export function Paw({ className = "icon" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
      <ellipse cx="12" cy="16.5" rx="5.2" ry="4.4" />
      <ellipse cx="5.2" cy="10.4" rx="2.1" ry="2.7" transform="rotate(-22 5.2 10.4)" />
      <ellipse cx="9.4" cy="6.4" rx="2.2" ry="2.8" transform="rotate(-8 9.4 6.4)" />
      <ellipse cx="14.6" cy="6.4" rx="2.2" ry="2.8" transform="rotate(8 14.6 6.4)" />
      <ellipse cx="18.8" cy="10.4" rx="2.1" ry="2.7" transform="rotate(22 18.8 10.4)" />
    </svg>
  );
}

export function Mascot({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true">
      <ellipse cx="15" cy="27" rx="8.5" ry="15" transform="rotate(24 15 27)" fill="#8b5a3c" />
      <ellipse cx="49" cy="27" rx="8.5" ry="15" transform="rotate(-24 49 27)" fill="#8b5a3c" />
      <circle cx="32" cy="35" r="19" fill="#f1bf86" />
      <ellipse cx="40" cy="27" rx="6" ry="5" fill="#e3a464" />
      <ellipse cx="32" cy="44" rx="11" ry="8" fill="#fde9cf" />
      <circle cx="25" cy="33" r="2.6" fill="#2e2219" />
      <circle cx="39" cy="33" r="2.6" fill="#2e2219" />
      <circle cx="25.9" cy="32.1" r="0.8" fill="#fff" />
      <circle cx="39.9" cy="32.1" r="0.8" fill="#fff" />
      <ellipse cx="32" cy="40" rx="4" ry="2.9" fill="#2e2219" />
      <path
        d="M32 42.5v2.5M27.5 45c2.2 2 6.8 2 9 0"
        stroke="#2e2219"
        strokeWidth="1.6"
        fill="none"
        strokeLinecap="round"
      />
      <path d="M30 46.5h4v2.6a2 2 0 0 1-4 0z" fill="#f08a8a" />
    </svg>
  );
}
