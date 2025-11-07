# Fraud Risk Patterns Knowledge Base

This document describes commonly recognized **fraud risk patterns** observed in payment and transaction systems.  
Each section provides a neutral definition, typical indicators, potential mechanisms, and suggested related features.  
The descriptions are compiled from public, industry, or regulatory sources and are intended to support model interpretability — not to assert definitive fraud.

---

## 1. Card-Not-Present (CNP) Transactions
**Definition:**  
Transactions where the payment card is not physically presented to the merchant, such as online, mobile, or phone orders.  
**Indicators (may include):**  
- Transaction channels without physical card verification (e-commerce, mail/phone orders).  
- Absent EMV chip usage (`use_chip = No`).  
- Mismatch between billing and shipping location.  
**Risk Mechanism:**  
CNP transactions depend solely on credential authenticity (card number, expiry, CVV). The lack of physical authentication may increase exposure to unauthorized use.  
**Related Features:** `use_chip`, `merchant_state`, `mcc`, `transaction_hour`, `amount`.  
**References:**  
- Stripe. “What Is Card-Not-Present Fraud?” (2023)  
- ACI Worldwide. “Understanding Card-Not-Present Fraud” (2023)  
- Visa. “CNP Fraud and Risk Controls Overview” (2022)

---

## 2. Enumeration / Card Testing
**Definition:**  
A pattern where attackers conduct numerous low-value authorization attempts to identify valid card credentials.  
**Indicators (may include):**  
- Multiple small or failed transactions within short time intervals.  
- Sequential or incremental card numbers tested from same device or IP.  
- Spikes in decline rates or unusual authorization velocity.  
**Risk Mechanism:**  
Enumeration helps attackers validate stolen card data before performing larger fraudulent purchases.  
**Related Features:** `amount`, `velocity_1h`, `merchant_id`, `ip_address`, `device_id`.  
**References:**  
- Visa. “Anti-Enumeration and Account Testing – Best Practices for Merchants” (2022)  
- Mastercard. “Account Testing Fraud Overview” (2023)  

---

## 3. Velocity Spikes
**Definition:**  
A short-term surge in the number or total value of transactions linked to a common key (card, account, IP, merchant, or device).  
**Indicators (may include):**  
- Rapid repetition of small-value purchases.  
- Significant deviation from historical activity patterns.  
**Risk Mechanism:**  
Fraudulent operations may occur in bursts before detection systems react. Monitoring transaction velocity is a core analytic control.  
**Related Features:** `velocity_*`, `transaction_hour`, `device_id`, `ip_address`.  
**References:**  
- U.S. Payments Forum. “Velocity Checks and Fraud Detection Practices” (2022)  
- Federal Reserve Payment Fraud Study (2023)

---

## 4. Account Takeover (ATO)
**Definition:**  
An event where a malicious actor gains unauthorized access to a legitimate customer account and conducts transactions or modifies details.  
**Indicators (may include):**  
- Login or transaction from unusual geography or device.  
- Recent password, email, or address change followed by purchases.  
- Increased decline or refund activity after account access changes.  
**Risk Mechanism:**  
Using valid credentials allows attackers to bypass some identity checks, leveraging existing trust to perform unauthorized actions.  
**Related Features:** `login_country`, `account_age`, `merchant_state`, `transaction_hour`.  
**References:**  
- Visa. “Account Takeover Fraud Best Practices” (2023)  
- Imperva. “What Is Account Takeover Fraud?” (2023)

---

## 5. High-Risk Merchant Category Codes (MCC)
**Definition:**  
Certain MCCs correspond to industries with higher chargeback or dispute rates, such as digital goods, gaming, or travel.  
**Indicators (may include):**  
- Merchant categories historically associated with elevated refund or dispute ratios.  
- MCCs listed as “high-risk” by card networks or acquirers.  
**Risk Mechanism:**  
These categories often involve fast-fulfilment, intangible goods, or cross-border delivery — factors correlated with increased dispute likelihood.  
**Related Features:** `mcc`, `merchant_id`, `amount`, `merchant_state`.  
**References:**  
- Visa Core Rules & Visa Product and Service Rules (2023)  
- Mastercard Chargeback Guide (2023)

---

## 6. Friendly Fraud / Chargeback Abuse
**Definition:**  
A situation in which a legitimate cardholder disputes a genuine transaction, intentionally or unintentionally, resulting in a chargeback.  
**Indicators (may include):**  
- Recurring billing, subscription, or digital-content models.  
- High dispute frequency from certain cardholders or merchants.  
**Risk Mechanism:**  
Abuse of the chargeback process may impose financial and operational losses; repeated disputes can signal potential misuse.  
**Related Features:** `chargeback_count_30d`, `amount`, `mcc`, `merchant_id`.  
**References:**  
- Mastercard. “Understanding Friendly Fraud” (2023)  
- Chargebacks911. “Chargeback Abuse and Prevention Strategies” (2023)

---

## 7. Triangulation Fraud
**Definition:**  
Fraudsters create an intermediary storefront, use stolen payment cards to buy real goods from legitimate merchants, and ship items to unsuspecting consumers.  
**Indicators (may include):**  
- Orders placed through third-party sellers or platforms; discrepancies between billing and delivery addresses.  
- Multiple distinct cards or accounts sending goods to a single address.  
**Risk Mechanism:**  
This pattern masks stolen-card usage behind legitimate-looking consumer orders.  
**Related Features:** `shipping_zip`, `billing_zip`, `merchant_id`, `amount`.  
**References:**  
- ACI Worldwide. “Triangulation Fraud Explained” (2022)  
- Fraud.com. “Common E-Commerce Fraud Patterns” (2023)

---

## 8. Synthetic Identity Fraud
**Definition:**  
Creation of a fictitious identity by combining real and fabricated personal information (e.g., real SSN with false name or date of birth).  
**Indicators (may include):**  
- Inconsistent identity attributes across applications.  
- Recently issued identification numbers; limited credit history.  
**Risk Mechanism:**  
Fraudsters build fake profiles to establish credit and later default or commit fraud.  
**Related Features:** `credit_score`, `acct_open_date`, `birth_year`, `num_credit_cards`.  
**References:**  
- FICO. “Synthetic Identity Fraud and Mitigation Strategies” (2023)  
- U.S. Federal Trade Commission. “Protecting Consumers from Synthetic Identity Fraud” (2022)

---

## 9. Refund Fraud
**Definition:**  
Exploitation of merchant refund processes to obtain unauthorized credits, goods, or monetary reimbursement.  
**Indicators (may include):**  
- Excessive or repetitive refund requests from same payment instrument.  
- Refunds to alternate accounts or new cards.  
**Risk Mechanism:**  
Fraudsters may claim non-delivery or use compromised accounts to trigger fraudulent refunds.  
**Related Features:** `refund_count_30d`, `amount`, `merchant_id`, `customer_id`.  
**References:**  
- Verifi. “Refund Fraud in Card-Not-Present Environments” (2023)  
- Chargebacks911. “Refund Fraud vs Friendly Fraud Differences” (2023)

---

## 10. Cross-Border / Geolocation Mismatch
**Definition:**  
Transactions where the cardholder, merchant, and device geographies differ unusually (e.g., card issued in one country, device in another).  
**Indicators (may include):**  
- Card-issuing country and merchant country mismatch.  
- IP location inconsistent with billing address.  
**Risk Mechanism:**  
Legitimate cross-border activity exists, but unusual or repeated mismatches can correlate with higher unauthorized use risk.  
**Related Features:** `merchant_country`, `billing_country`, `ip_country`, `transaction_hour`.  
**References:**  
- Mastercard. “Cross-Border Fraud Mitigation Guidelines” (2023)  
- Visa. “International Transaction Risk Controls” (2023)

---

## [NOTE] How to Use This Document
When generating explanations through the Fraud-Explanation Chatbot:  
- Combine one **feature definition** (from `fraud_knowledge_base_features.csv`) and one **risk pattern** section from this document to support reasoning.  
- Use cautious, evidence-based language such as “may indicate”, “consistent with”, or “potentially related to”.  
- Always cite sources in square brackets, e.g., `[fraud_knowledge_base_patterns.md::Velocity Spikes]`.  
- If no relevant evidence is retrieved, state “insufficient evidence from knowledge base.”  
- This document provides contextual knowledge; it does not determine outcomes.

---

## References (Consolidated)
1. Stripe. *What Is Card-Not-Present Fraud?* (2023) – [https://stripe.com/resources/more/what-is-card-not-present-fraud-what-businesses-need-to-know](https://stripe.com/resources/more/what-is-card-not-present-fraud-what-businesses-need-to-know)  
2. Visa. *Anti-Enumeration and Account Testing Best Practices for Merchants.* (2022) – [https://usa.visa.com/dam/VCOM/global/support-legal/documents/anti-enumeration-and-account-testing-best-practices-merchant.pdf](https://usa.visa.com/dam/VCOM/global/support-legal/documents/anti-enumeration-and-account-testing-best-practices-merchant.pdf)  
3. Mastercard. *Chargeback Guide and Friendly Fraud Overview.* (2023)  
4. U.S. Payments Forum. *Velocity Checks and Fraud Detection Practices.* (2022)  
5. ACI Worldwide. *Understanding Card-Not-Present and Triangulation Fraud.* (2022–2023)  
6. FICO. *Synthetic Identity Fraud and Mitigation Strategies.* (2023)  
7. Federal Trade Commission (FTC). *Protecting Consumers from Synthetic Identity Fraud.* (2022)  
8. Imperva. *Account Takeover Fraud Explained.* (2023)  
9. Verifi. *Refund Fraud in Card-Not-Present Environments.* (2023)  
10. Chargebacks911. *Refund Fraud and Dispute Prevention.* (2023)  
11. Visa Core Rules & Mastercard Cross-Border Fraud Mitigation Guidelines (2023)

---
