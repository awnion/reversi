//! Log lines in the format of `log_event` in `python/train/loop.py`:
//! `HH:MM:SS [component] message key=value ...`, one line per event.

use std::fmt::Display;
use std::io::Write;

/// Local wall-clock time as `HH:MM:SS`.
fn stamp() -> String {
    // SAFETY: `time` and `localtime_r` only write into the provided structs.
    unsafe {
        let now = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&now, &mut tm).is_null() {
            return "??:??:??".to_owned();
        }
        format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

/// Print one event; `fields` are `(key, value)` pairs.
pub fn event(component: &str, message: &str, fields: &[(&str, &dyn Display)]) {
    let mut line = format!("{} [{component}] {message}", stamp());
    for (key, value) in fields {
        line.push_str(&format!(" {key}={value}"));
    }
    line.push('\n');
    // One write per line, so lines from different threads do not interleave.
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(line.as_bytes());
    let _ = out.flush();
}

/// `log_event` for the self-play component.
#[macro_export]
macro_rules! log_selfplay {
    ($message:expr $(, $key:ident = $value:expr)* $(,)?) => {
        $crate::log::event(
            "selfplay",
            $message,
            &[$((stringify!($key), &$value as &dyn ::std::fmt::Display)),*],
        )
    };
}

/// `log_event` with an explicit component.
#[macro_export]
macro_rules! log_event {
    ($component:expr, $message:expr $(, $key:ident = $value:expr)* $(,)?) => {
        $crate::log::event(
            $component,
            $message,
            &[$((stringify!($key), &$value as &dyn ::std::fmt::Display)),*],
        )
    };
}
