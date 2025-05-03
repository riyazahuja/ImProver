/-- Recursion principle on even and odd numbers: if we have `P 0`, and for all `i : ℕ` we can
extend from `P i` to both `P (2 * i)` and `P (2 * i + 1)`, then we have `P n` for all `n : ℕ`.
This is nothing more than a wrapper around `Nat.binaryRec`, to avoid having to switch to
dealing with `bit0` and `bit1`. -/
@[elab_as_elim]
def evenOddRec {P : ℕ → Sort*} (h0 : P 0) (h_even : ∀ n, P n → P (2 * n))
    (h_odd : ∀ n, P n → P (2 * n + 1)) (n : ℕ) : P n :=
  binaryRec h0 (fun
    | false, i, hi => (h_even i hi : P (2 * i))
    | true, i, hi => (h_odd i hi : P (2 * i + 1))) n


@[simp]
theorem evenOddRec_zero {P : ℕ → Sort*} (h0 : P 0) (h_even : ∀ i, P i → P (2 * i))
    (h_odd : ∀ i, P i → P (2 * i + 1)) : evenOddRec h0 h_even h_odd 0 = h0 :=
  binaryRec_zero _ _


@[simp]
theorem evenOddRec_even {P : ℕ → Sort*} (h0 : P 0) (h_even : ∀ i, P i → P (2 * i))
    (h_odd : ∀ i, P i → P (2 * i + 1)) (H : h_even 0 h0 = h0) (n : ℕ) :
    (2 * n).evenOddRec h0 h_even h_odd = h_even n (evenOddRec h0 h_even h_odd n) := by
  /-
    P : Nat → Sort u_1
    h0 : P 0
    h_even : (i : Nat) → P i → P (HMul.hMul 2 i)
    h_odd : (i : Nat) → P i → P (HAdd.hAdd (HMul.hMul 2 i) 1)
    H : Eq (h_even 0 h0) h0
    n : Nat
    ⊢ Eq (Nat.evenOddRec h0 h_even h_odd (HMul.hMul 2 n)) (h_even n (Nat.evenOddRe …
  -/
  apply binaryRec_eq false n
  /-
    P : Nat → Sort u_1
    h0 : P 0
    h_even : (i : Nat) → P i → P (HMul.hMul 2 i)
    h_odd : (i : Nat) → P i → P (HAdd.hAdd (HMul.hMul 2 i) 1)
    H : Eq (h_even 0 h0) h0
    n : Nat
    ⊢ Or (Eq (Nat.evenOddRec.match_1 (fun x x_1 x_2 => (fun x => P x) (Nat.bit x x …
  -/
  simp [H]
  /-
    🎉 no goals
  -/


@[simp]
theorem evenOddRec_odd {P : ℕ → Sort*} (h0 : P 0) (h_even : ∀ i, P i → P (2 * i))
    (h_odd : ∀ i, P i → P (2 * i + 1)) (H : h_even 0 h0 = h0) (n : ℕ) :
    (2 * n + 1).evenOddRec h0 h_even h_odd = h_odd n (evenOddRec h0 h_even h_odd n) := by
  /-
    P : Nat → Sort u_1
    h0 : P 0
    h_even : (i : Nat) → P i → P (HMul.hMul 2 i)
    h_odd : (i : Nat) → P i → P (HAdd.hAdd (HMul.hMul 2 i) 1)
    H : Eq (h_even 0 h0) h0
    n : Nat
    ⊢ Eq (Nat.evenOddRec h0 h_even h_odd (HAdd.hAdd (HMul.hMul 2 n) 1)) (h_odd n ( …
  -/
  apply binaryRec_eq true n
  /-
    P : Nat → Sort u_1
    h0 : P 0
    h_even : (i : Nat) → P i → P (HMul.hMul 2 i)
    h_odd : (i : Nat) → P i → P (HAdd.hAdd (HMul.hMul 2 i) 1)
    H : Eq (h_even 0 h0) h0
    n : Nat
    ⊢ Or (Eq (Nat.evenOddRec.match_1 (fun x x_1 x_2 => (fun x => P x) (Nat.bit x x …
  -/
  simp [H]
  /-
    🎉 no goals
  -/


/-- Strong recursion principle on even and odd numbers: if for all `i : ℕ` we can prove `P (2 * i)`
from `P j` for all `j < 2 * i` and we can prove `P (2 * i + 1)` from `P j` for all `j < 2 * i + 1`,
then we have `P n` for all `n : ℕ`. -/
@[elab_as_elim]
noncomputable def evenOddStrongRec {P : ℕ → Sort*}
    (h_even : ∀ n : ℕ, (∀ k < 2 * n, P k) → P (2 * n))
    (h_odd : ∀ n : ℕ, (∀ k < 2 * n + 1, P k) → P (2 * n + 1)) (n : ℕ) : P n :=
  n.strongRecOn fun m ih => m.even_or_odd'.choose_spec.by_cases
    (fun h => h.symm ▸ h_even m.even_or_odd'.choose <| h ▸ ih)
    (fun h => h.symm ▸ h_odd m.even_or_odd'.choose <| h ▸ ih)


