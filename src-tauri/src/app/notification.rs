//! Native notification delivery with a real click callback.
//!
//! `tauri-plugin-notification` is fire-and-forget on desktop: `show()` hands the
//! notification to `notify_rust` and drops the handle, so nothing ever tells the
//! page which notification the user clicked. Because Pake replaces the page's
//! `window.Notification` wholesale, that also kills the page's own click
//! routing: Slack never jumps to the conversation the notification came from.
//!
//! macOS therefore delivers notifications itself through
//! `NSUserNotificationCenter` with a delegate and routes the click back to the
//! originating webview. Other platforms keep the plugin path, and the page falls
//! back to the focus heuristic in `inject/event.js`.

use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, WebviewWindow};

/// Ids are minted by the packaged (untrusted) page and echoed into a native
/// notification identifier and a webview `eval`, so keep them short and opaque.
const MAX_NOTIFICATION_ID_LEN: usize = 64;

/// How long the same message coming from a different window counts as a repeat.
/// Windows are driven by the same server event, so they fire near-simultaneously;
/// this only needs to absorb ordinary scheduling jitter, and staying tight keeps
/// a genuinely repeated message from being swallowed.
const DEDUPE_WINDOW: Duration = Duration::from_secs(1);
const MAX_TRACKED_MESSAGES: usize = 64;

/// One incoming message, and the notification each window raised for it.
struct MessageGroup {
    key: String,
    /// When the message was first seen, for the repeat window only. Membership
    /// itself outlives that: a click can land minutes after delivery.
    at: Instant,
    /// `(window label, notification id)` in arrival order. The first entry is
    /// the notification that was actually delivered; the rest were suppressed.
    members: Vec<(String, String)>,
}

/// Collapses the copies of one message that `--multi-window` produces, and
/// remembers them so a click can be routed to a window of our choosing.
///
/// Every Pake window runs its own instance of the site, so a single incoming
/// message raises one notification per window, all identical and all competing
/// for the same click. Only a *different* window repeating the same title+body
/// counts as a duplicate, so a site legitimately repeating a message within one
/// window still gets every notification.
#[derive(Default)]
struct MessageRegistry {
    groups: Vec<MessageGroup>,
}

impl MessageRegistry {
    /// Records a notification and reports whether it should be suppressed
    /// because another window already raised the same message.
    fn record(&mut self, window_label: &str, id: &str, key: String, now: Instant) -> bool {
        let member = (window_label.to_string(), id.to_string());

        // Only a group inside the repeat window that this window has not
        // contributed to yet absorbs the notification. A window repeating a
        // message to itself therefore starts a new group and is delivered.
        let existing = self.groups.iter_mut().find(|group| {
            group.key == key
                && now.duration_since(group.at) < DEDUPE_WINDOW
                && !group.members.iter().any(|(w, _)| w == window_label)
        });

        if let Some(group) = existing {
            group.members.push(member);
            return true;
        }

        if self.groups.len() >= MAX_TRACKED_MESSAGES {
            self.groups.remove(0);
        }
        self.groups.push(MessageGroup {
            key,
            at: now,
            members: vec![member],
        });
        false
    }

    /// The notification `preferred_label` raised for the same message.
    ///
    /// `None` when that window never raised it -- it may have been opened after
    /// the message arrived, or still be loading -- in which case the caller
    /// keeps the click on the window that raised the clicked notification
    /// rather than dropping it.
    fn sibling_in(
        &self,
        window_label: &str,
        id: &str,
        preferred_label: &str,
    ) -> Option<(String, String)> {
        if window_label == preferred_label {
            return None;
        }
        self.groups
            .iter()
            .find(|group| {
                group
                    .members
                    .iter()
                    .any(|(w, i)| w == window_label && i == id)
            })?
            .members
            .iter()
            .find(|(w, _)| w == preferred_label)
            .cloned()
    }
}

static REGISTRY: Mutex<MessageRegistry> = Mutex::new(MessageRegistry { groups: Vec::new() });

/// Unit separator keeps a title ending in the body's prefix from colliding.
fn dedupe_key(title: &str, body: &str) -> String {
    format!("{title}\u{1f}{body}")
}

// A poisoned lock only means some earlier caller panicked mid-update; the worst
// case here is a stale entry, never a reason to drop or misroute a notification.
fn is_cross_window_repeat(window_label: &str, id: &str, title: &str, body: &str) -> bool {
    let key = dedupe_key(title, body);
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.record(window_label, id, key, Instant::now())
}

#[cfg(target_os = "macos")]
fn sibling_in(window_label: &str, id: &str, preferred_label: &str) -> Option<(String, String)> {
    let registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.sibling_in(window_label, id, preferred_label)
}

#[derive(serde::Deserialize)]
pub struct NotificationParams {
    id: String,
    title: String,
    body: String,
    icon: String,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationOutcome {
    /// True when this platform reports the click back to the page. The page
    /// keeps its focus-based fallback disabled in that case, so an ordinary app
    /// switch never fires a phantom click on the newest notification.
    native_click: bool,
    /// True when another window already raised this exact message, so nothing
    /// was shown here. The page keeps the notification addressable -- a click
    /// on the window that did show it can be rerouted to this one -- but must
    /// not report a show event or count it towards the badge, which would
    /// otherwise multiply by the number of windows.
    suppressed: bool,
}

fn validate_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > MAX_NOTIFICATION_ID_LEN {
        return Err(format!(
            "Notification id must be 1-{MAX_NOTIFICATION_ID_LEN} characters"
        ));
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("Notification id must be alphanumeric, '-' or '_'".to_string());
    }
    Ok(())
}

/// Set up the platform click callback. Runs on the main thread during setup so
/// the availability probe is a plain synchronous check later on.
pub fn init_native_click(app: &AppHandle) {
    #[cfg(target_os = "macos")]
    macos::init(app);
    #[cfg(not(target_os = "macos"))]
    let _ = app;
}

pub fn send(
    app: &AppHandle,
    window: &WebviewWindow,
    params: &NotificationParams,
) -> Result<NotificationOutcome, String> {
    validate_id(&params.id)?;

    if is_cross_window_repeat(window.label(), &params.id, &params.title, &params.body) {
        return Ok(NotificationOutcome {
            native_click: false,
            suppressed: true,
        });
    }

    #[cfg(target_os = "macos")]
    if macos::deliver(app, window.label(), params)? {
        return Ok(NotificationOutcome {
            native_click: true,
            suppressed: false,
        });
    }

    use tauri_plugin_notification::NotificationExt;
    app.notification()
        .builder()
        .title(&params.title)
        .body(&params.body)
        .icon(&params.icon)
        .show()
        .map_err(|e| format!("Failed to show notification: {e}"))?;

    Ok(NotificationOutcome {
        native_click: false,
        suppressed: false,
    })
}

/// Dismiss a notification the page closed itself.
///
/// Pages close notifications once the user has read the message elsewhere; with
/// multiple Pake windows every window's copy of the site raises its own
/// notification for the same message, so honouring `close()` is what clears the
/// duplicates. Silently a no-op where the platform gives us no handle.
pub fn close(app: &AppHandle, window: &WebviewWindow, id: &str) -> Result<(), String> {
    validate_id(id)?;

    #[cfg(target_os = "macos")]
    macos::remove(app, window.label(), id)?;
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, window);
    }

    Ok(())
}

/// Hand a notification click to the right window.
///
/// With `--multi-window` every window runs its own instance of the site, so the
/// notification could have come from any of them. The click is steered to the
/// tab the user is actually looking at -- resolved at click time, since they may
/// have switched tabs since it arrived -- but only when that tab also raised the
/// message. Routing means dispatching to the site's own Notification object,
/// and only the window that created one has the conversation to jump to.
///
/// A visible tab with no notification for the message has almost always gone
/// quiet precisely because it is already showing that conversation: chat sites
/// do not notify about a message the user can see. The message is then already
/// in front of the user, so pulling a background tab over the one they are
/// working in would be the wrong answer, and this does nothing instead. The
/// cost is that a visible tab which genuinely missed the message (still
/// loading, say) makes the click look inert.
#[cfg(target_os = "macos")]
fn dispatch_click(app: &AppHandle, window_label: &str, id: &str) {
    use tauri::Manager;

    let visible = macos::preferred_click_target(app, window_label);

    if let Some(visible_label) = visible.as_deref().filter(|label| *label != window_label) {
        if let Some((label, sibling_id)) = sibling_in(window_label, id, visible_label) {
            if let Some(window) = app.get_webview_window(&label) {
                reveal_and_deliver(&window, &sibling_id);
                return;
            }
        }

        // Leave the user on the tab they are working in.
        if app.get_webview_window(visible_label).is_some() {
            return;
        }
    }

    // Single window, or the notification belongs to the visible tab already.
    if let Some(window) = app.get_webview_window(window_label) {
        reveal_and_deliver(&window, id);
    }
}

/// A hidden or minimized window is exactly the case where a notification click
/// matters most, so this goes through the same show + `reapply_window_icon` +
/// focus sequence as every other hidden-to-visible path (#1323).
#[cfg(target_os = "macos")]
fn reveal_and_deliver(window: &WebviewWindow, id: &str) {
    let _ = window.unminimize();
    let _ = window.show();
    crate::app::window::reapply_window_icon(window);
    let _ = window.set_focus();

    // `id` passed `validate_id`, so it cannot break out of the string literal.
    let _ = window.eval(format!(
        "window.__pakeNotificationClick && window.__pakeNotificationClick('{id}')"
    ));
}

#[cfg(target_os = "macos")]
mod macos {
    // The whole NSUserNotification family is deprecated in favour of
    // UserNotifications.framework, which needs a provisioned bundle and a
    // permission prompt that a webpage wrapper cannot meaningfully ask for.
    // This is also the API the notification plugin already reaches through
    // notify-rust, so staying on it keeps notification appearance unchanged.
    #![allow(deprecated)]

    use super::{dispatch_click, NotificationParams};
    use objc2::rc::Retained;
    use objc2::runtime::{AnyClass, ProtocolObject};
    use objc2::{define_class, msg_send, DefinedClass, MainThreadMarker, MainThreadOnly};
    use objc2_app_kit::NSWindow;
    use objc2_foundation::{
        NSObject, NSObjectProtocol, NSString, NSUserNotification, NSUserNotificationCenter,
        NSUserNotificationCenterDelegate,
    };
    use std::cell::OnceCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tauri::AppHandle;

    static NATIVE_CLICK_READY: AtomicBool = AtomicBool::new(false);

    thread_local! {
        static DELEGATE: OnceCell<Retained<Delegate>> = const { OnceCell::new() };
    }

    struct DelegateIvars {
        app: AppHandle,
    }

    define_class!(
        // SAFETY:
        // - NSObject has no subclassing requirements.
        // - Delegate does not implement Drop.
        #[unsafe(super(NSObject))]
        #[thread_kind = MainThreadOnly]
        #[name = "PakeUserNotificationDelegate"]
        #[ivars = DelegateIvars]
        struct Delegate;

        unsafe impl NSObjectProtocol for Delegate {}

        unsafe impl NSUserNotificationCenterDelegate for Delegate {
            #[unsafe(method(userNotificationCenter:didActivateNotification:))]
            fn did_activate(
                &self,
                center: &NSUserNotificationCenter,
                notification: &NSUserNotification,
            ) {
                // Read the identifier before removing: keep the two steps
                // independent of any ordering assumption about the removal.
                let raw = notification.identifier().map(|s| s.to_string());

                // NSUserNotificationCenter keeps an activated notification in
                // Notification Center; without this it piles up after every click.
                center.removeDeliveredNotification(notification);

                let Some(identifier) = raw else {
                    return;
                };
                // `rsplit_once` because a popup window label comes from the
                // page's `window.open` name and may itself contain '|', while
                // the id never can.
                let Some((label, id)) = identifier.rsplit_once('|') else {
                    return;
                };
                dispatch_click(&self.ivars().app, label, id);
            }

            // Show the banner even when Pake is frontmost: the window that owns
            // the conversation may still be hidden or in the background, and the
            // click is what routes the page there.
            #[unsafe(method(userNotificationCenter:shouldPresentNotification:))]
            fn should_present(
                &self,
                _center: &NSUserNotificationCenter,
                _notification: &NSUserNotification,
            ) -> bool {
                true
            }
        }
    );

    impl Delegate {
        fn new(mtm: MainThreadMarker, app: AppHandle) -> Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(DelegateIvars { app });
            unsafe { msg_send![super(this), init] }
        }
    }

    /// `defaultUserNotificationCenter` is nil for a process without a bundle
    /// identifier (`pnpm run dev` runs the bare binary), and the class itself
    /// disappears if a future macOS finally drops the deprecated API. Probe once
    /// here rather than risking a nil dereference on every notification.
    fn default_center(mtm: MainThreadMarker) -> Option<Retained<NSUserNotificationCenter>> {
        let _ = mtm;
        let class = AnyClass::get(c"NSUserNotificationCenter")?;
        unsafe { msg_send![class, defaultUserNotificationCenter] }
    }

    pub fn init(app: &AppHandle) {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        let Some(center) = default_center(mtm) else {
            return;
        };

        DELEGATE.with(|cell| {
            let delegate = cell.get_or_init(|| Delegate::new(mtm, app.clone()));
            unsafe { center.setDelegate(Some(ProtocolObject::from_ref(&**delegate))) };
        });
        NATIVE_CLICK_READY.store(true, Ordering::SeqCst);
    }

    /// Label of the window the user is currently looking at.
    ///
    /// `NSWindowTabGroup::selectedWindow` is the tab on screen, which is what a
    /// notification click should land in, and it is read at click time so
    /// switching or reordering tabs after the notification arrived is accounted
    /// for. It does not depend on Pake being frontmost, so it still resolves
    /// while the user is in another app.
    ///
    /// Windows outside a tab group fall back to whichever one holds focus.
    /// `None` means the caller keeps its existing behaviour.
    pub fn preferred_click_target(app: &AppHandle, origin_label: &str) -> Option<String> {
        use tauri::Manager;

        MainThreadMarker::new()?;

        let windows = app.webview_windows();
        let label_of = |target: *const NSWindow| {
            windows.iter().find_map(|(label, window)| {
                let ptr = window.ns_window().ok()? as *const NSWindow;
                (ptr == target).then(|| label.clone())
            })
        };

        let origin_ns = windows.get(origin_label)?.ns_window().ok()? as *mut NSWindow;
        // SAFETY: Tauri hands back the window's live NSWindow, and this runs on
        // the main thread, where AppKit window state may be read.
        if let Some(group) = unsafe { (*origin_ns).tabGroup() } {
            if let Some(selected) = group.selectedWindow() {
                return label_of(Retained::as_ptr(&selected));
            }
        }

        windows
            .iter()
            .find_map(|(label, window)| window.is_focused().ok()?.then(|| label.clone()))
    }

    /// Returns whether the notification was handed to the native center. `false`
    /// means the caller should fall back to the plugin path.
    pub fn deliver(
        app: &AppHandle,
        window_label: &str,
        params: &NotificationParams,
    ) -> Result<bool, String> {
        if !NATIVE_CLICK_READY.load(Ordering::SeqCst) {
            return Ok(false);
        }

        let identifier = format!("{window_label}|{}", params.id);
        let title = params.title.clone();
        let body = params.body.clone();

        app.run_on_main_thread(move || {
            let Some(mtm) = MainThreadMarker::new() else {
                return;
            };
            let Some(center) = default_center(mtm) else {
                return;
            };

            let notification = NSUserNotification::new();
            notification.setTitle(Some(&NSString::from_str(&title)));
            notification.setInformativeText(Some(&NSString::from_str(&body)));
            notification.setIdentifier(Some(&NSString::from_str(&identifier)));
            center.deliverNotification(&notification);
        })
        .map_err(|e| format!("Failed to dispatch notification: {e}"))?;

        Ok(true)
    }

    /// Withdraw an already-delivered notification. Matching by identifier keeps
    /// one window's `close()` from clearing another window's copy.
    pub fn remove(app: &AppHandle, window_label: &str, id: &str) -> Result<(), String> {
        if !NATIVE_CLICK_READY.load(Ordering::SeqCst) {
            return Ok(());
        }

        let identifier = format!("{window_label}|{id}");

        app.run_on_main_thread(move || {
            let Some(mtm) = MainThreadMarker::new() else {
                return;
            };
            let Some(center) = default_center(mtm) else {
                return;
            };

            for notification in center.deliveredNotifications().iter() {
                let matches = notification
                    .identifier()
                    .is_some_and(|current| current.to_string() == identifier);
                if matches {
                    center.removeDeliveredNotification(&notification);
                }
            }
        })
        .map_err(|e| format!("Failed to dispatch notification removal: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{dedupe_key, validate_id, MessageRegistry, DEDUPE_WINDOW};
    use std::time::{Duration, Instant};

    /// `record` returns "was suppressed", so `false` means it got delivered.
    fn deliver(registry: &mut MessageRegistry, window: &str, id: &str, at: Instant) -> bool {
        !registry.record(window, id, dedupe_key("Ann", "hi"), at)
    }

    #[test]
    fn accepts_generated_ids() {
        assert!(validate_id("pake-12-a1b2c3d4").is_ok());
        assert!(validate_id("A_b-9").is_ok());
    }

    #[test]
    fn rejects_ids_that_could_escape_an_eval_literal() {
        for id in ["", "a'b", "a\\b", "a|b", "a b", "a\nb", "a;b"] {
            assert!(validate_id(id).is_err(), "expected {id:?} to be rejected");
        }
        assert!(validate_id(&"a".repeat(65)).is_err());
    }

    #[test]
    fn collapses_the_same_message_from_another_window() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();

        assert!(deliver(&mut registry, "pake", "a", now));
        assert!(!deliver(&mut registry, "pake-1", "b", now));
        assert!(!deliver(&mut registry, "pake-2", "c", now));
    }

    #[test]
    fn keeps_a_message_the_same_window_repeats() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();

        assert!(deliver(&mut registry, "pake", "a", now));
        assert!(deliver(&mut registry, "pake", "b", now));
    }

    #[test]
    fn keeps_a_different_message_from_another_window() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();

        assert!(!registry.record("pake", "a", dedupe_key("Ann", "hi"), now));
        assert!(!registry.record("pake-1", "b", dedupe_key("Bo", "hi"), now));
        assert!(!registry.record("pake-1", "c", dedupe_key("Ann", "bye"), now));
    }

    #[test]
    fn stops_collapsing_once_the_window_has_passed() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();
        let later = now + DEDUPE_WINDOW + Duration::from_millis(1);

        assert!(deliver(&mut registry, "pake", "a", now));
        assert!(deliver(&mut registry, "pake-1", "b", later));
    }

    #[test]
    fn separates_title_and_body_so_they_cannot_run_together() {
        assert_ne!(dedupe_key("ab", "c"), dedupe_key("a", "bc"));
    }

    #[test]
    fn reroutes_a_click_to_the_preferred_window() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();
        deliver(&mut registry, "pake-1", "delivered", now);
        deliver(&mut registry, "pake", "suppressed", now);

        // Clicking pake-1's banner hands the click to pake's own notification.
        assert_eq!(
            registry.sibling_in("pake-1", "delivered", "pake"),
            Some(("pake".to_string(), "suppressed".to_string()))
        );
    }

    #[test]
    fn does_not_reroute_when_already_on_the_preferred_window() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();
        deliver(&mut registry, "pake", "delivered", now);
        deliver(&mut registry, "pake-1", "suppressed", now);

        assert_eq!(registry.sibling_in("pake", "delivered", "pake"), None);
    }

    #[test]
    fn does_not_reroute_when_the_preferred_window_missed_the_message() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();
        deliver(&mut registry, "pake-1", "delivered", now);

        // pake-2 opened later and never raised this message, so the click has
        // to stay with pake-1 rather than vanish.
        assert_eq!(registry.sibling_in("pake-1", "delivered", "pake-2"), None);
    }

    #[test]
    fn keeps_routing_after_the_dedupe_window_expires() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();
        deliver(&mut registry, "pake-1", "delivered", now);
        deliver(&mut registry, "pake", "suppressed", now);

        // Membership must outlive the repeat window: a click can land minutes
        // after the notification was delivered.
        let _ = deliver(
            &mut registry,
            "pake-1",
            "later",
            now + DEDUPE_WINDOW + Duration::from_secs(60),
        );
        assert_eq!(
            registry.sibling_in("pake-1", "delivered", "pake"),
            Some(("pake".to_string(), "suppressed".to_string()))
        );
    }

    #[test]
    fn does_not_confuse_two_messages_with_the_same_text() {
        let mut registry = MessageRegistry::default();
        let now = Instant::now();
        deliver(&mut registry, "pake-1", "first-a", now);
        deliver(&mut registry, "pake", "first-b", now);

        let later = now + DEDUPE_WINDOW + Duration::from_millis(1);
        deliver(&mut registry, "pake-1", "second-a", later);
        deliver(&mut registry, "pake", "second-b", later);

        assert_eq!(
            registry.sibling_in("pake-1", "second-a", "pake"),
            Some(("pake".to_string(), "second-b".to_string()))
        );
    }
}
