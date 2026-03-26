import { useTranslation } from 'react-i18next';
import './LanguageSwitcher.css';

const languages = [
  { code: 'de', label: 'DE', flag: '🇩🇪' },
  { code: 'en', label: 'EN', flag: '🇬🇧' },
  { code: 'fr', label: 'FR', flag: '🇫🇷' },
];

function LanguageSwitcher() {
  const { i18n } = useTranslation();

  return (
    <div className="language-switcher">
      {languages.map(({ code, label, flag }) => (
        <button
          key={code}
          className={`lang-button ${i18n.resolvedLanguage === code ? 'active' : ''}`}
          onClick={() => i18n.changeLanguage(code)}
          aria-label={label}
        >
          <span className="lang-flag">{flag}</span>
          <span className="lang-label">{label}</span>
        </button>
      ))}
    </div>
  );
}

export default LanguageSwitcher;
