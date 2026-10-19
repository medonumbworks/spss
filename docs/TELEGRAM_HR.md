# Spajanje tvog Telegram računa (besplatno, oko 10 minuta)

CryptoPulse bez ikakve prijave već čita javne kripto kanale na Telegramu. Kad spojiš svoj račun,
besplatni "sentiment hub" na GitHub Actions čita i **sve grupe i kanale u kojima jesi**, plus javne
kripto grupe po imenu, i šalje ti sažetak tržišta u **Saved Messages**.

Iz privatnih grupa objavljuju se samo brojke po kriptovaluti (broj spominjanja, raspoloženje).
Tekstovi poruka i imena privatnih grupa nikad se ne objavljuju niti zapisuju.

## 1. Telegram API ključ (besplatan)

1. Otvori https://my.telegram.org i prijavi se svojim brojem mobitela.
2. Klikni **API development tools** i ispuni kratki obrazac (App title: `CryptoPulse`, Platform: `Other`).
3. Zapiši **api_id** (broj) i **api_hash** (dugi niz znakova).

## 2. Jednokratna prijava (stvara "session")

Najlakše u pregledniku, bez instalacije: na stranici repozitorija
https://github.com/medonumbworks/spss klikni **Code → Codespaces → Create codespace** (besplatno).
U terminalu koji se otvori upiši:

```bash
pip install telethon
python tools/sentiment_hub/login.py
```

Upiši api_id, api_hash, broj mobitela, kod koji ti Telegram pošalje i lozinku ako imaš 2FA.
Skripta ispiše dugi niz znakova: to je tvoj **TELEGRAM_SESSION**.

**Taj niz je potpuni pristup tvom računu. Nikad ga ne dijeli i ne stavljaj u kod.**
Možeš ga opozvati u Telegramu: Settings → Devices → CryptoPulse → Terminate.

## 3. Spremi tajne u GitHub

Otvori https://github.com/medonumbworks/spss/settings/secrets/actions i dodaj tri tajne
(**New repository secret**):

| Ime | Vrijednost |
|-----|-----------|
| `TELEGRAM_API_ID` | broj iz koraka 1 |
| `TELEGRAM_API_HASH` | niz iz koraka 1 |
| `TELEGRAM_SESSION` | niz iz koraka 2 |

## 4. Uključi automatske poslove

Zakazani poslovi na GitHubu rade samo na glavnoj grani (`main`). Kad spojiš ovu granu u `main`:

* **sentiment-hub** se pokreće svaki sat i objavljuje `feed.json` koji aplikacija preuzima;
* **telegram-digest** tri puta dnevno šalje sažetak u tvoj Saved Messages (na hrvatskom).

Ručno pokretanje: **Actions → sentiment-hub → Run workflow** (ili telegram-digest).

## Bez osobnog računa: Telegram bot

Ako ne želiš spajati račun, sažetak može slati bot:
1. U Telegramu otvori **@BotFather**, pošalji `/newbot` i zapiši token.
2. Pošalji svom botu bilo koju poruku, zatim otvori `https://api.telegram.org/bot<TOKEN>/getUpdates`
   i zapiši `chat.id`.
3. Dodaj tajne `TELEGRAM_BOT_TOKEN` i `TELEGRAM_CHAT_ID`.

Bot može slati sažetke, ali ne može čitati tuđe grupe; za čitanje grupa treba tvoj račun.
