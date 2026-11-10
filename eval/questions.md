# Answer Key - Nimbus Tech Knowledge Assistant

Use this to test the system after every change. For each question, record:
(a) was the right chunk retrieved?  (b) was the answer correct?  (c) was the citation correct?

## Type 1: Single fact, one document

1. Question: How many days of paid annual leave does a full-time employee get?
   Expected answer: 24 days per calendar year
   Found in: leave_policy.docx (section 1)

2. Question: What is the minimum password length?
   Expected answer: 14 characters
   Found in: it_security_policy.pdf, page 1

3. Question: How many vacation days can I roll over into next year, and when do they expire?
   Expected answer: Up to 5 days; they expire on 31 March
   Found in: leave_policy.docx (section 1)
   Why it is tricky: the question says "vacation" and "roll over"; the policy says "annual leave" and "carried forward".

4. Question: How quickly must a security incident be reported?
   Expected answer: Within 2 hours of discovery
   Found in: it_security_policy.pdf, page 3 (tests page citation)

## Type 2: Table / CSV lookups

5. Question: What is the daily hotel limit for international travel, and does it need approval?
   Expected answer: 260 USD per day; yes, manager approval is needed before booking
   Found in: expense_policy.csv (row: Hotel - international)

6. Question: How many weeks of parental leave does a primary caregiver get?
   Expected answer: 16 weeks paid (after 6 months of service)
   Found in: leave_policy.docx (table in section 3)

## Type 3: Needs two documents

7. Question: My laptop was lost. What must I do, and who do I call for a replacement?
   Expected answer: Report it within 2 hours (security@nimbustech.example or hotline ext. 4400); then call the IT helpdesk, ext. 4200, for a replacement
   Found in: it_security_policy.pdf (page 3) AND company_handbook.txt ("Who to contact")

## Type 4: Follow-up pair (tests conversation memory)

8a. Question: What is the sick leave policy?
    Expected answer: 12 days of paid sick leave per year; no carry-forward
    Found in: leave_policy.docx (section 2)

8b. Follow-up: Do I need a doctor's note for it?
    Expected answer: Yes, a medical certificate if the absence is longer than 3 consecutive days
    Found in: leave_policy.docx (section 2)
    Why it is tricky: "it" only makes sense from question 8a.

## Type 5: NOT in the documents (system must refuse, not invent)

9. Question: What is the stock option vesting schedule?
   Expected answer: Not in the documents - the assistant should say it could not find sufficient evidence.

10. Question: How much is the monthly remote-work internet stipend?
    Expected answer: Not in the documents - the assistant should say it could not find sufficient evidence.
    Why it is tricky: the handbook discusses hybrid working, so related text exists, but no stipend is mentioned.

## Scoring sheet (fill in as you test)

| # | Right chunk retrieved? | Answer correct? | Citation correct? | Notes |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8a | | | | |
| 8b | | | | |
| 9 | | | | |
| 10 | | | | |
