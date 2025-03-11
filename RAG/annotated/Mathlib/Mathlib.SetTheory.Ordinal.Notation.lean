set_option genSizeOfSpec false in
/-- Recursive definition of an ordinal notation. `zero` denotes the ordinal 0, and `oadd e n a` is
intended to refer to `ω ^ e * n + a`. For this to be a valid Cantor normal form, we must have the
exponents decrease to the right, but we can't state this condition until we've defined `repr`, so we
make it a separate definition `NF`. -/
inductive ONote : Type
  | zero : ONote
  | oadd : ONote → ℕ+ → ONote → ONote
  deriving DecidableEq


compile_inductive% ONote


/-- Notation for 0 -/
instance : Zero ONote :=
  ⟨zero⟩


@[simp]
theorem zero_def : zero = 0 :=
  rfl


instance : Inhabited ONote :=
  ⟨0⟩


/-- Notation for 1 -/
instance : One ONote :=
  ⟨oadd 0 1 0⟩


/-- Notation for ω -/
def omega : ONote :=
  oadd 1 1 0


/-- The ordinal denoted by a notation -/
@[simp]
noncomputable def repr : ONote → Ordinal.{0}
  | 0 => 0
  | oadd e n a => ω ^ repr e * n + repr a


/-- Print `ω^s*n`, omitting `s` if `e = 0` or `e = 1`, and omitting `n` if `n = 1` -/
private def toString_aux (e : ONote) (n : ℕ) (s : String) : String :=
  if e = 0 then toString n
  else (if e = 1 then "ω" else "ω^(" ++ s ++ ")") ++ if n = 1 then "" else "*" ++ toString n


/-- Print an ordinal notation -/
def toString : ONote → String
  | zero => "0"
  | oadd e n 0 => toString_aux e n (toString e)
  | oadd e n a => toString_aux e n (toString e) ++ " + " ++ toString a


open Lean in
/-- Print an ordinal notation -/
def repr' (prec : ℕ) : ONote → Format
  | zero => "0"
  | oadd e n a =>
    Repr.addAppParen
      ("oadd " ++ (repr' max_prec e) ++ " " ++ Nat.repr (n : ℕ) ++ " " ++ (repr' max_prec a))
      prec


instance : ToString ONote :=
  ⟨toString⟩


instance : Repr ONote where
  reprPrec o prec := repr' prec o


instance : Preorder ONote where
  le x y := repr x ≤ repr y
  lt x y := repr x < repr y
  le_refl _ := @le_refl Ordinal _ _
  le_trans _ _ _ := @le_trans Ordinal _ _ _ _
  lt_iff_le_not_le _ _ := @lt_iff_le_not_le Ordinal _ _ _


theorem lt_def {x y : ONote} : x < y ↔ repr x < repr y :=
  Iff.rfl


theorem le_def {x y : ONote} : x ≤ y ↔ repr x ≤ repr y :=
  Iff.rfl


instance : WellFoundedRelation ONote :=
  ⟨(· < ·), InvImage.wf repr Ordinal.lt_wf⟩


/-- Convert a `Nat` into an ordinal -/
@[coe]
def ofNat : ℕ → ONote
  | 0 => 0
  | Nat.succ n => oadd 0 n.succPNat 0

-- Porting note (https://github.com/leanprover-community/mathlib4/pull/11467): during the port we marked these lemmas with `@[eqns]`
-- to emulate the old Lean 3 behaviour.


@[simp] theorem ofNat_zero : ofNat 0 = 0 :=
  rfl


@[simp] theorem ofNat_succ (n) : ofNat (Nat.succ n) = oadd 0 n.succPNat 0 :=
  rfl


instance nat (n : ℕ) : OfNat ONote n where
  ofNat := ofNat n


@[simp 1200]
theorem ofNat_one : ofNat 1 = 1 :=
  rfl


@[simp]
                                                      /-
                                                        n : Nat
                                                        ⊢ Eq (↑n).repr ↑n
                                                      -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
theorem repr_ofNat (n : ℕ) : repr (ofNat n) = n := by cases n <;> simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem repr_one : repr (ofNat 1) = (1 : ℕ) := repr_ofNat 1


theorem omega0_le_oadd (e n a) : ω ^ repr e ≤ repr (oadd e n a) := by
  /-
    e : ONote
    n : PNat
    a : ONote
    ⊢ LE.le (HPow.hPow Ordinal.omega0 e.repr) (e.oadd n a).repr
  -/
  refine le_trans ?_ (le_add_right _ _)
  /-
    e : ONote
    n : PNat
    a : ONote
    ⊢ LE.le (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul (HPow.hPow Ordinal.omega0 …
  -/
  simpa using (Ordinal.mul_le_mul_iff_left <| opow_pos (repr e) omega0_pos).2 (Nat.cast_le.2 n.2)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-30")]
alias omega_le_oadd := omega0_le_oadd


theorem oadd_pos (e n a) : 0 < oadd e n a :=
  @lt_of_lt_of_le _ _ _ (ω ^ repr e) _ (opow_pos (repr e) omega0_pos) (omega0_le_oadd e n a)


/-- Comparison of ordinal notations:

`ω ^ e₁ * n₁ + a₁` is less than `ω ^ e₂ * n₂ + a₂` when either `e₁ < e₂`, or `e₁ = e₂` and
`n₁ < n₂`, or `e₁ = e₂`, `n₁ = n₂`, and `a₁ < a₂`. -/
def cmp : ONote → ONote → Ordering
  | 0, 0 => Ordering.eq
  | _, 0 => Ordering.gt
  | 0, _ => Ordering.lt
  | _o₁@(oadd e₁ n₁ a₁), _o₂@(oadd e₂ n₂ a₂) =>
    (cmp e₁ e₂).then <| (_root_.cmp (n₁ : ℕ) n₂).then (cmp a₁ a₂)


theorem eq_of_cmp_eq : ∀ {o₁ o₂}, cmp o₁ o₂ = Ordering.eq → o₁ = o₂
  | 0, 0, _ => rfl
                           /-
                             e : ONote
                             n : PNat
                             a : ONote
                             h : Eq ((e.oadd n a).cmp 0) Ordering.eq
                             ⊢ Eq (e.oadd n a) 0
                           -/
  | oadd e n a, 0, h => by injection h
                           /-
                             🎉 no goals
                           -/
                           /-
                             e : ONote
                             n : PNat
                             a : ONote
                             h : Eq (ONote.cmp 0 (e.oadd n a)) Ordering.eq
                             ⊢ Eq 0 (e.oadd n a)
                           -/
  | 0, oadd e n a, h => by injection h
                           /-
                             🎉 no goals
                           -/
  | oadd e₁ n₁ a₁, oadd e₂ n₂ a₂, h => by
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h : Eq ((e₁.oadd n₁ a₁).cmp (e₂.oadd n₂ a₂)) Ordering.eq
      ⊢ Eq (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)
    -/
    revert h; simp only [cmp]
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      ⊢ Eq ((e₁.cmp e₂).then ((_root_.cmp ↑n₁ ↑n₂).then (a₁.cmp a₂))) Ordering.eq →  …
    -/
                                         /-
                                           🎉 no goals
                                         -/
    cases h₁ : cmp e₁ e₂ <;> intro h <;> try cases h
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case eq
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : Eq (e₁.cmp e₂) Ordering.eq
      h : Eq (Ordering.eq.then ((_root_.cmp ↑n₁ ↑n₂).then (a₁.cmp a₂))) Ordering.eq
      ⊢ Eq (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)
    -/
    obtain rfl := eq_of_cmp_eq h₁
    /-
      case eq
      e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      n₂ : PNat
      a₂ : ONote
      h : Eq (Ordering.eq.then ((_root_.cmp ↑n₁ ↑n₂).then (a₁.cmp a₂))) Ordering.eq
      h₁ : Eq (e₁.cmp e₁) Ordering.eq
      ⊢ Eq (e₁.oadd n₁ a₁) (e₁.oadd n₂ a₂)
    -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    revert h; cases h₂ : _root_.cmp (n₁ : ℕ) n₂ <;> intro h <;> try cases h
                                                                /-
                                                                  🎉 no goals
                                                                -/
    /-
      case eq.eq
      e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : Eq (e₁.cmp e₁) Ordering.eq
      h₂ : Eq (_root_.cmp ↑n₁ ↑n₂) Ordering.eq
      h : Eq (Ordering.eq.then (Ordering.eq.then (a₁.cmp a₂))) Ordering.eq
      ⊢ Eq (e₁.oadd n₁ a₁) (e₁.oadd n₂ a₂)
    -/
    obtain rfl := eq_of_cmp_eq h
    /-
      case eq.eq
      e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      n₂ : PNat
      h₁ : Eq (e₁.cmp e₁) Ordering.eq
      h₂ : Eq (_root_.cmp ↑n₁ ↑n₂) Ordering.eq
      h : Eq (Ordering.eq.then (Ordering.eq.then (a₁.cmp a₁))) Ordering.eq
      ⊢ Eq (e₁.oadd n₁ a₁) (e₁.oadd n₂ a₁)
    -/
    rw [_root_.cmp, cmpUsing_eq_eq, not_lt, not_lt, ← le_antisymm_iff] at h₂
    /-
      case eq.eq
      e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      n₂ : PNat
      h₁ : Eq (e₁.cmp e₁) Ordering.eq
      h₂ : Eq ↑n₂ ↑n₁
      h : Eq (Ordering.eq.then (Ordering.eq.then (a₁.cmp a₁))) Ordering.eq
      ⊢ Eq (e₁.oadd n₁ a₁) (e₁.oadd n₂ a₁)
    -/
    obtain rfl := Subtype.eq h₂
    /-
      case eq.eq
      e₁ a₁ : ONote
      n₂ : PNat
      h₁ : Eq (e₁.cmp e₁) Ordering.eq
      h : Eq (Ordering.eq.then (Ordering.eq.then (a₁.cmp a₁))) Ordering.eq
      h₂ : Eq ↑n₂ ↑n₂
      ⊢ Eq (e₁.oadd n₂ a₁) (e₁.oadd n₂ a₁)
    -/
    simp
    /-
      🎉 no goals
    -/


protected theorem zero_lt_one : (0 : ONote) < 1 := by
  simp only [lt_def, repr, opow_zero, Nat.succPNat_coe, Nat.cast_one, mul_one, add_zero,
    zero_lt_one]


/-- `NFBelow o b` says that `o` is a normal form ordinal notation satisfying `repr o < ω ^ b`. -/
inductive NFBelow : ONote → Ordinal.{0} → Prop
  | zero {b} : NFBelow 0 b
  | oadd' {e n a eb b} : NFBelow e eb → NFBelow a (repr e) → repr e < b → NFBelow (oadd e n a) b


/-- A normal form ordinal notation has the form

`ω ^ a₁ * n₁ + ω ^ a₂ * n₂ + ⋯ + ω ^ aₖ * nₖ`

where `a₁ > a₂ > ⋯ > aₖ` and all the `aᵢ` are also in normal form.

We will essentially only be interested in normal form ordinal notations, but to avoid complicating
the algorithms, we define everything over general ordinal notations and only prove correctness with
normal form as an invariant. -/
class NF (o : ONote) : Prop where
  out : Exists (NFBelow o)


instance NF.zero : NF 0 :=
  ⟨⟨0, NFBelow.zero⟩⟩


theorem NFBelow.oadd {e n a b} : NF e → NFBelow a (repr e) → repr e < b → NFBelow (oadd e n a) b
  | ⟨⟨_, h⟩⟩ => NFBelow.oadd' h


theorem NFBelow.fst {e n a b} (h : NFBelow (ONote.oadd e n a) b) : NF e := by
  /-
    e : ONote
    n : PNat
    a : ONote
    b : Ordinal.{0}
    h : (e.oadd n a).NFBelow b
    ⊢ e.NF
  -/
  cases' h with _ _ _ _ eb _ h₁ h₂ h₃; exact ⟨⟨_, h₁⟩⟩
                                       /-
                                         🎉 no goals
                                       -/


theorem NF.fst {e n a} : NF (oadd e n a) → NF e
  | ⟨⟨_, h⟩⟩ => h.fst


theorem NFBelow.snd {e n a b} (h : NFBelow (ONote.oadd e n a) b) : NFBelow a (repr e) := by
  /-
    e : ONote
    n : PNat
    a : ONote
    b : Ordinal.{0}
    h : (e.oadd n a).NFBelow b
    ⊢ a.NFBelow e.repr
  -/
  cases' h with _ _ _ _ eb _ h₁ h₂ h₃; exact h₂
                                       /-
                                         🎉 no goals
                                       -/


theorem NF.snd' {e n a} : NF (oadd e n a) → NFBelow a (repr e)
  | ⟨⟨_, h⟩⟩ => h.snd


theorem NF.snd {e n a} (h : NF (oadd e n a)) : NF a :=
  ⟨⟨_, h.snd'⟩⟩


theorem NF.oadd {e a} (h₁ : NF e) (n) (h₂ : NFBelow a (repr e)) : NF (oadd e n a) :=
  ⟨⟨_, NFBelow.oadd h₁ h₂ (lt_succ _)⟩⟩


instance NF.oadd_zero (e n) [h : NF e] : NF (ONote.oadd e n 0) :=
  h.oadd _ NFBelow.zero


theorem NFBelow.lt {e n a b} (h : NFBelow (ONote.oadd e n a) b) : repr e < b := by
  /-
    e : ONote
    n : PNat
    a : ONote
    b : Ordinal.{0}
    h : (e.oadd n a).NFBelow b
    ⊢ LT.lt e.repr b
  -/
  cases' h with _ _ _ _ eb _ h₁ h₂ h₃; exact h₃
                                       /-
                                         🎉 no goals
                                       -/


theorem NFBelow_zero : ∀ {o}, NFBelow o 0 ↔ o = 0
  | 0 => ⟨fun _ => rfl, fun _ => NFBelow.zero⟩
  | oadd _ _ _ =>
    ⟨fun h => (not_le_of_lt h.lt).elim (Ordinal.zero_le _), fun e => e.symm ▸ NFBelow.zero⟩


theorem NF.zero_of_zero {e n a} (h : NF (ONote.oadd e n a)) (e0 : e = 0) : a = 0 := by
  /-
    e : ONote
    n : PNat
    a : ONote
    h : (e.oadd n a).NF
    e0 : Eq e 0
    ⊢ Eq a 0
  -/
  simpa [e0, NFBelow_zero] using h.snd'
  /-
    🎉 no goals
  -/


theorem NFBelow.repr_lt {o b} (h : NFBelow o b) : repr o < ω ^ b := by
  induction h with
  | zero => exact opow_pos _ omega0_pos
  | oadd' _ _ h₃ _ IH =>
    rw [repr]
    apply ((add_lt_add_iff_left _).2 IH).trans_le
    rw [← mul_succ]
    apply (mul_le_mul_left' (succ_le_of_lt (nat_lt_omega0 _)) _).trans
    rw [← opow_succ]
    exact opow_le_opow_right omega0_pos (succ_le_of_lt h₃)


theorem NFBelow.mono {o b₁ b₂} (bb : b₁ ≤ b₂) (h : NFBelow o b₁) : NFBelow o b₂ := by
  induction h with
  | zero => exact zero
  | oadd' h₁ h₂ h₃ _ _ => constructor; exacts [h₁, h₂, lt_of_lt_of_le h₃ bb]


theorem NF.below_of_lt {e n a b} (H : repr e < b) :
    NF (ONote.oadd e n a) → NFBelow (ONote.oadd e n a) b
                     /-
                       e : ONote
                       n : PNat
                       a : ONote
                       b : Ordinal.{0}
                       H : LT.lt e.repr b
                       b' : Ordinal.{0}
                       h : (e.oadd n a).NFBelow b'
                       ⊢ (e.oadd n a).NFBelow b
                     -/
  | ⟨⟨b', h⟩⟩ => by (cases' h with _ _ _ _ eb _ h₁ h₂ h₃; exact NFBelow.oadd' h₁ h₂ H)
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem NF.below_of_lt' : ∀ {o b}, repr o < ω ^ b → NF o → NFBelow o b
  | 0, _, _, _ => NFBelow.zero
  | ONote.oadd _ _ _, _, H, h =>
    h.below_of_lt <|
      (opow_lt_opow_iff_right one_lt_omega0).1 <| lt_of_le_of_lt (omega0_le_oadd _ _ _) H


theorem nfBelow_ofNat : ∀ n, NFBelow (ofNat n) 1
  | 0 => NFBelow.zero
  | Nat.succ _ => NFBelow.oadd NF.zero NFBelow.zero zero_lt_one


instance nf_ofNat (n) : NF (ofNat n) :=
  ⟨⟨_, nfBelow_ofNat n⟩⟩


                             /-
                               ⊢ ONote.NF 1
                             -/
instance nf_one : NF 1 := by rw [← ofNat_one]; infer_instance
                                               /-
                                                 🎉 no goals
                                               -/


theorem oadd_lt_oadd_1 {e₁ n₁ o₁ e₂ n₂ o₂} (h₁ : NF (oadd e₁ n₁ o₁)) (h : e₁ < e₂) :
    oadd e₁ n₁ o₁ < oadd e₂ n₂ o₂ :=
  @lt_of_lt_of_le _ _ (repr (oadd e₁ n₁ o₁)) _ _
    (NF.below_of_lt h h₁).repr_lt (omega0_le_oadd e₂ n₂ o₂)


theorem oadd_lt_oadd_2 {e o₁ o₂ : ONote} {n₁ n₂ : ℕ+} (h₁ : NF (oadd e n₁ o₁)) (h : (n₁ : ℕ) < n₂) :
    oadd e n₁ o₁ < oadd e n₂ o₂ := by
  /-
    e o₁ o₂ : ONote
    n₁ n₂ : PNat
    h₁ : (e.oadd n₁ o₁).NF
    h : LT.lt ↑n₁ ↑n₂
    ⊢ LT.lt (e.oadd n₁ o₁) (e.oadd n₂ o₂)
  -/
  simp only [lt_def, repr]
  /-
    e o₁ o₂ : ONote
    n₁ n₂ : PNat
    h₁ : (e.oadd n₁ o₁).NF
    h : LT.lt ↑n₁ ↑n₂
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n₁) o₁.repr) …
  -/
  refine lt_of_lt_of_le ((add_lt_add_iff_left _).2 h₁.snd'.repr_lt) (le_trans ?_ (le_add_right _ _))
  /-
    e o₁ o₂ : ONote
    n₁ n₂ : PNat
    h₁ : (e.oadd n₁ o₁).NF
    h : LT.lt ↑n₁ ↑n₂
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n₁) (HPow.hP …
  -/
  rwa [← mul_succ,Ordinal.mul_le_mul_iff_left (opow_pos _ omega0_pos), succ_le_iff, Nat.cast_lt]
  /-
    🎉 no goals
  -/


theorem oadd_lt_oadd_3 {e n a₁ a₂} (h : a₁ < a₂) : oadd e n a₁ < oadd e n a₂ := by
  /-
    e : ONote
    n : PNat
    a₁ a₂ : ONote
    h : LT.lt a₁ a₂
    ⊢ LT.lt (e.oadd n a₁) (e.oadd n a₂)
  -/
  rw [lt_def]; unfold repr
  /-
    e : ONote
    n : PNat
    a₁ a₂ : ONote
    h : LT.lt a₁ a₂
    ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a₁.repr)  …
  -/
  exact @add_lt_add_left _ _ _ _ (repr a₁) _ h _
  /-
    🎉 no goals
  -/


theorem cmp_compares : ∀ (a b : ONote) [NF a] [NF b], (cmp a b).Compares a b
  | 0, 0, _, _ => rfl
  | oadd _ _ _, 0, _, _ => oadd_pos _ _ _
  | 0, oadd _ _ _, _, _ => oadd_pos _ _ _
  | o₁@(oadd e₁ n₁ a₁), o₂@(oadd e₂ n₂ a₂), h₁, h₂ => by -- TODO: golf
    /-
      o₁ e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      h✝¹ : Eq o₁ (e₁.oadd n₁ a₁)
      o₂ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h✝ : Eq o₂ (e₂.oadd n₂ a₂)
      h₁ : (namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).NF
      h₂ : (namedPattern o₂ (e₂.oadd n₂ a₂) h✝).NF
      ⊢ ((namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).cmp (namedPattern o₂ (e₂.oadd n₂ a₂)  …
    -/
    rw [cmp]
    /-
      o₁ e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      h✝¹ : Eq o₁ (e₁.oadd n₁ a₁)
      o₂ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h✝ : Eq o₂ (e₂.oadd n₂ a₂)
      h₁ : (namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).NF
      h₂ : (namedPattern o₂ (e₂.oadd n₂ a₂) h✝).NF
      ⊢ ((e₁.cmp e₂).then ((_root_.cmp ↑n₁ ↑n₂).then (a₁.cmp a₂))).Compares (namedPa …
    -/
    have IHe := @cmp_compares _ _ h₁.fst h₂.fst
    /-
      o₁ e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      h✝¹ : Eq o₁ (e₁.oadd n₁ a₁)
      o₂ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h✝ : Eq o₂ (e₂.oadd n₂ a₂)
      h₁ : (namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).NF
      h₂ : (namedPattern o₂ (e₂.oadd n₂ a₂) h✝).NF
      IHe : (e₁.cmp e₂).Compares e₁ e₂
      ⊢ ((e₁.cmp e₂).then ((_root_.cmp ↑n₁ ↑n₂).then (a₁.cmp a₂))).Compares (namedPa …
    -/
    simp only [Ordering.Compares, gt_iff_lt] at IHe; revert IHe
    /-
      o₁ e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      h✝¹ : Eq o₁ (e₁.oadd n₁ a₁)
      o₂ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h✝ : Eq o₂ (e₂.oadd n₂ a₂)
      h₁ : (namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).NF
      h₂ : (namedPattern o₂ (e₂.oadd n₂ a₂) h✝).NF
      ⊢ (Ordering.Compares.match_1 (fun x x x => Prop) (e₁.cmp e₂) e₁ e₂ (fun a b => …
    -/
    cases cmp e₁ e₂
    /-
      case lt
      o₁ e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      h✝¹ : Eq o₁ (e₁.oadd n₁ a₁)
      o₂ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h✝ : Eq o₂ (e₂.oadd n₂ a₂)
      h₁ : (namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).NF
      h₂ : (namedPattern o₂ (e₂.oadd n₂ a₂) h✝).NF
      ⊢ (Ordering.Compares.match_1 (fun x x x => Prop) Ordering.lt e₁ e₂ (fun a b => …
    -/
    case lt => intro IHe; exact oadd_lt_oadd_1 h₁ IHe
    /-
      case eq
      o₁ e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      h✝¹ : Eq o₁ (e₁.oadd n₁ a₁)
      o₂ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h✝ : Eq o₂ (e₂.oadd n₂ a₂)
      h₁ : (namedPattern o₁ (e₁.oadd n₁ a₁) h✝¹).NF
      h₂ : (namedPattern o₂ (e₂.oadd n₂ a₂) h✝).NF
      ⊢ (Ordering.Compares.match_1 (fun x x x => Prop) Ordering.eq e₁ e₂ (fun a b => …
    -/
    case gt => intro IHe; exact oadd_lt_oadd_1 h₂ IHe
    case eq =>
      intro IHe; dsimp at IHe; subst IHe
      unfold _root_.cmp; cases nh : cmpUsing (· < ·) (n₁ : ℕ) n₂ <;>
      rw [cmpUsing, ite_eq_iff, not_lt] at nh
      case lt =>
        cases' nh with nh nh
        · exact oadd_lt_oadd_2 h₁ nh.left
        · rw [ite_eq_iff] at nh; cases' nh.right with nh nh <;> cases nh <;> contradiction
      case gt =>
        cases' nh with nh nh
        · cases nh; contradiction
        · cases' nh with _ nh
          rw [ite_eq_iff] at nh; cases' nh with nh nh
          · exact oadd_lt_oadd_2 h₂ nh.left
          · cases nh; contradiction
      cases' nh with nh nh
      · cases nh; contradiction
      cases' nh with nhl nhr
      rw [ite_eq_iff] at nhr
      cases' nhr with nhr nhr
      · cases nhr; contradiction
      obtain rfl := Subtype.eq (nhl.eq_of_not_lt nhr.1)
      have IHa := @cmp_compares _ _ h₁.snd h₂.snd
      revert IHa; cases cmp a₁ a₂ <;> intro IHa <;> dsimp at IHa
      case lt => exact oadd_lt_oadd_3 IHa
      case gt => exact oadd_lt_oadd_3 IHa
      subst IHa; exact rfl


theorem repr_inj {a b} [NF a] [NF b] : repr a = repr b ↔ a = b :=
  ⟨fun e => match cmp a b, cmp_compares a b with
    | Ordering.lt, (h : repr a < repr b) => (ne_of_lt h e).elim
    | Ordering.gt, (h : repr a > repr b)=> (ne_of_gt h e).elim
    | Ordering.eq, h => h,
    congr_arg _⟩


theorem NF.of_dvd_omega0_opow {b e n a} (h : NF (ONote.oadd e n a))
    (d : ω ^ b ∣ repr (ONote.oadd e n a)) :
    b ≤ repr e ∧ ω ^ b ∣ repr a := by
  /-
    b : Ordinal.{0}
    e : ONote
    n : PNat
    a : ONote
    h : (e.oadd n a).NF
    d : Dvd.dvd (HPow.hPow Ordinal.omega0 b) (e.oadd n a).repr
    ⊢ And (LE.le b e.repr) (Dvd.dvd (HPow.hPow Ordinal.omega0 b) a.repr)
  -/
  have := mt repr_inj.1 (fun h => by injection h : ONote.oadd e n a ≠ 0)
  /-
    b : Ordinal.{0}
    e : ONote
    n : PNat
    a : ONote
    h : (e.oadd n a).NF
    d : Dvd.dvd (HPow.hPow Ordinal.omega0 b) (e.oadd n a).repr
    this : Not (Eq (e.oadd n a).repr (ONote.repr 0))
    ⊢ And (LE.le b e.repr) (Dvd.dvd (HPow.hPow Ordinal.omega0 b) a.repr)
  -/
  have L := le_of_not_lt fun l => not_le_of_lt (h.below_of_lt l).repr_lt (le_of_dvd this d)
  /-
    b : Ordinal.{0}
    e : ONote
    n : PNat
    a : ONote
    h : (e.oadd n a).NF
    d : Dvd.dvd (HPow.hPow Ordinal.omega0 b) (e.oadd n a).repr
    this : Not (Eq (e.oadd n a).repr (ONote.repr 0))
    L : LE.le b e.repr
    ⊢ And (LE.le b e.repr) (Dvd.dvd (HPow.hPow Ordinal.omega0 b) a.repr)
  -/
  simp only [repr] at d
  /-
    b : Ordinal.{0}
    e : ONote
    n : PNat
    a : ONote
    h : (e.oadd n a).NF
    d : Dvd.dvd (HPow.hPow Ordinal.omega0 b) (HAdd.hAdd (HMul.hMul (HPow.hPow Ordi …
    this : Not (Eq (e.oadd n a).repr (ONote.repr 0))
    L : LE.le b e.repr
    ⊢ And (LE.le b e.repr) (Dvd.dvd (HPow.hPow Ordinal.omega0 b) a.repr)
  -/
  exact ⟨L, (dvd_add_iff <| (opow_dvd_opow _ L).mul_right _).1 d⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-30")]
alias NF.of_dvd_omega_opow := NF.of_dvd_omega0_opow


theorem NF.of_dvd_omega0 {e n a} (h : NF (ONote.oadd e n a)) :
    ω ∣ repr (ONote.oadd e n a) → repr e ≠ 0 ∧ ω ∣ repr a := by
   /-
     e : ONote
     n : PNat
     a : ONote
     h : (e.oadd n a).NF
     ⊢ Dvd.dvd Ordinal.omega0 (e.oadd n a).repr → And (Ne e.repr 0) (Dvd.dvd Ordina …
   -/
  (rw [← opow_one ω, ← one_le_iff_ne_zero]; exact h.of_dvd_omega0_opow)
                                            /-
                                              🎉 no goals
                                            -/


@[deprecated (since := "2024-09-30")]
alias NF.of_dvd_omega := NF.of_dvd_omega0


/-- `TopBelow b o` asserts that the largest exponent in `o`, if it exists, is less than `b`. This is
an auxiliary definition for decidability of `NF`. -/
def TopBelow (b : ONote) : ONote → Prop
  | 0 => True
  | oadd e _ _ => cmp e b = Ordering.lt


instance decidableTopBelow : DecidableRel TopBelow := by
  /-
    ⊢ DecidableRel ONote.TopBelow
  -/
  intro b o
  /-
    b o : ONote
    ⊢ Decidable (b.TopBelow o)
  -/
                                 /-
                                   🎉 no goals
                                 -/
  cases o <;> delta TopBelow <;> infer_instance
                                 /-
                                   🎉 no goals
                                 -/


theorem nfBelow_iff_topBelow {b} [NF b] : ∀ {o}, NFBelow o (repr b) ↔ NF o ∧ TopBelow b o
  | 0 => ⟨fun h => ⟨⟨⟨_, h⟩⟩, trivial⟩, fun _ => NFBelow.zero⟩
  | oadd _ _ _ =>
    ⟨fun h => ⟨⟨⟨_, h⟩⟩, (@cmp_compares _ b h.fst _).eq_lt.2 h.lt⟩, fun ⟨h₁, h₂⟩ =>
      h₁.below_of_lt <| (@cmp_compares _ b h₁.fst _).eq_lt.1 h₂⟩


instance decidableNF : DecidablePred NF
  | 0 => isTrue NF.zero
  | oadd e n a => by
    /-
      e : ONote
      n : PNat
      a : ONote
      ⊢ Decidable (e.oadd n a).NF
    -/
    have := decidableNF e
    /-
      e : ONote
      n : PNat
      a : ONote
      this : Decidable e.NF
      ⊢ Decidable (e.oadd n a).NF
    -/
    have := decidableNF a
    /-
      e : ONote
      n : PNat
      a : ONote
      this✝ : Decidable e.NF
      this : Decidable a.NF
      ⊢ Decidable (e.oadd n a).NF
    -/
    apply decidable_of_iff (NF e ∧ NF a ∧ TopBelow e a)
    /-
      case h
      e : ONote
      n : PNat
      a : ONote
      this✝ : Decidable e.NF
      this : Decidable a.NF
      ⊢ Iff (And e.NF (And a.NF (e.TopBelow a))) (e.oadd n a).NF
    -/
    rw [← and_congr_right fun h => @nfBelow_iff_topBelow _ h _]
    /-
      case h
      e : ONote
      n : PNat
      a : ONote
      this✝ : Decidable e.NF
      this : Decidable a.NF
      ⊢ Iff (And e.NF (a.NFBelow e.repr)) (e.oadd n a).NF
    -/
    exact ⟨fun ⟨h₁, h₂⟩ => NF.oadd h₁ n h₂, fun h => ⟨h.fst, h.snd'⟩⟩
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `add` -/
def addAux (e : ONote) (n : ℕ+) (o : ONote) : ONote :=
    match o with
    | 0 => oadd e n 0
    | o'@(oadd e' n' a') =>
      match cmp e e' with
      | Ordering.lt => o'
      | Ordering.eq => oadd e (n + n') a'
      | Ordering.gt => oadd e n o'


/-- Addition of ordinal notations (correct only for normal input) -/
def add : ONote → ONote → ONote
  | 0, o => o
  | oadd e n a, o => addAux e n (add a o)


instance : Add ONote :=
  ⟨add⟩


@[simp]
theorem zero_add (o : ONote) : 0 + o = o :=
  rfl


theorem oadd_add (e n a o) : oadd e n a + o = addAux e n (a + o) :=
  rfl


/-- Subtraction of ordinal notations (correct only for normal input) -/
def sub : ONote → ONote → ONote
  | 0, _ => 0
  | o, 0 => o
  | o₁@(oadd e₁ n₁ a₁), oadd e₂ n₂ a₂ =>
    match cmp e₁ e₂ with
    | Ordering.lt => 0
    | Ordering.gt => o₁
    | Ordering.eq =>
      match (n₁ : ℕ) - n₂ with
      | 0 => if n₁ = n₂ then sub a₁ a₂ else 0
      | Nat.succ k => oadd e₁ k.succPNat a₁


instance : Sub ONote :=
  ⟨sub⟩


theorem add_nfBelow {b} : ∀ {o₁ o₂}, NFBelow o₁ b → NFBelow o₂ b → NFBelow (o₁ + o₂) b
  | 0, _, _, h₂ => h₂
  | oadd e n a, o, h₁, h₂ => by
    /-
      b : Ordinal.{0}
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NFBelow b
      h₂ : o.NFBelow b
      ⊢ (HAdd.hAdd (e.oadd n a) o).NFBelow b
    -/
    have h' := add_nfBelow (h₁.snd.mono <| le_of_lt h₁.lt) h₂
    /-
      b : Ordinal.{0}
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NFBelow b
      h₂ : o.NFBelow b
      h' : (HAdd.hAdd a o).NFBelow b
      ⊢ (HAdd.hAdd (e.oadd n a) o).NFBelow b
    -/
    simp only [oadd_add]; revert h'; cases' a + o with e' n' a' <;> intro h'
      /-
        case zero
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        h' : ONote.zero.NFBelow b
        ⊢ (e.addAux n ONote.zero).NFBelow b
      -/
    · exact NFBelow.oadd h₁.fst NFBelow.zero h₁.lt
      /-
        🎉 no goals
      -/
    /-
      case oadd
      b : Ordinal.{0}
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NFBelow b
      h₂ : o.NFBelow b
      e' : ONote
      n' : PNat
      a' : ONote
      h' : (e'.oadd n' a').NFBelow b
      ⊢ (e.addAux n (e'.oadd n' a')).NFBelow b
    -/
    have : ((e.cmp e').Compares e e') := @cmp_compares _ _ h₁.fst h'.fst
    /-
      case oadd
      b : Ordinal.{0}
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NFBelow b
      h₂ : o.NFBelow b
      e' : ONote
      n' : PNat
      a' : ONote
      h' : (e'.oadd n' a').NFBelow b
      this : (e.cmp e').Compares e e'
      ⊢ (e.addAux n (e'.oadd n' a')).NFBelow b
    -/
    cases h : cmp e e' <;> dsimp [addAux] <;> simp only [h]
      /-
        case oadd.lt
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        e' : ONote
        n' : PNat
        a' : ONote
        h' : (e'.oadd n' a').NFBelow b
        this : (e.cmp e').Compares e e'
        h : Eq (e.cmp e') Ordering.lt
        ⊢ (e'.oadd n' a').NFBelow b
      -/
    · exact h'
      /-
        🎉 no goals
      -/
      /-
        case oadd.eq
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        e' : ONote
        n' : PNat
        a' : ONote
        h' : (e'.oadd n' a').NFBelow b
        this : (e.cmp e').Compares e e'
        h : Eq (e.cmp e') Ordering.eq
        ⊢ (e.oadd (HAdd.hAdd n n') a').NFBelow b
      -/
    · simp only [h] at this
      /-
        case oadd.eq
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        e' : ONote
        n' : PNat
        a' : ONote
        h' : (e'.oadd n' a').NFBelow b
        h : Eq (e.cmp e') Ordering.eq
        this : Ordering.eq.Compares e e'
        ⊢ (e.oadd (HAdd.hAdd n n') a').NFBelow b
      -/
      subst e'
      /-
        case oadd.eq
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        n' : PNat
        a' : ONote
        h' : (e.oadd n' a').NFBelow b
        h : Eq (e.cmp e) Ordering.eq
        ⊢ (e.oadd (HAdd.hAdd n n') a').NFBelow b
      -/
      exact NFBelow.oadd h'.fst h'.snd h'.lt
      /-
        🎉 no goals
      -/
      /-
        case oadd.gt
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        e' : ONote
        n' : PNat
        a' : ONote
        h' : (e'.oadd n' a').NFBelow b
        this : (e.cmp e').Compares e e'
        h : Eq (e.cmp e') Ordering.gt
        ⊢ (e.oadd n (e'.oadd n' a')).NFBelow b
      -/
    · simp only [h] at this
      /-
        case oadd.gt
        b : Ordinal.{0}
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NFBelow b
        h₂ : o.NFBelow b
        e' : ONote
        n' : PNat
        a' : ONote
        h' : (e'.oadd n' a').NFBelow b
        h : Eq (e.cmp e') Ordering.gt
        this : Ordering.gt.Compares e e'
        ⊢ (e.oadd n (e'.oadd n' a')).NFBelow b
      -/
      exact NFBelow.oadd h₁.fst (NF.below_of_lt this ⟨⟨_, h'⟩⟩) h₁.lt
      /-
        🎉 no goals
      -/


instance add_nf (o₁ o₂) : ∀ [NF o₁] [NF o₂], NF (o₁ + o₂)
  | ⟨⟨b₁, h₁⟩⟩, ⟨⟨b₂, h₂⟩⟩ =>
    ⟨(le_total b₁ b₂).elim (fun h => ⟨b₂, add_nfBelow (h₁.mono h) h₂⟩) fun h =>
        ⟨b₁, add_nfBelow h₁ (h₂.mono h)⟩⟩


@[simp]
theorem repr_add : ∀ (o₁ o₂) [NF o₁] [NF o₂], repr (o₁ + o₂) = repr o₁ + repr o₂
                     /-
                       o : ONote
                       x✝¹ : ONote.NF 0
                       x✝ : o.NF
                       ⊢ Eq (HAdd.hAdd 0 o).repr (HAdd.hAdd (ONote.repr 0) o.repr)
                     -/
  | 0, o, _, _ => by simp
                     /-
                       🎉 no goals
                     -/
  | oadd e n a, o, h₁, h₂ => by
    /-
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      ⊢ Eq (HAdd.hAdd (e.oadd n a) o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
    -/
    haveI := h₁.snd; have h' := repr_add a o
    /-
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this : a.NF
      h' : Eq (HAdd.hAdd a o).repr (HAdd.hAdd a.repr o.repr)
      ⊢ Eq (HAdd.hAdd (e.oadd n a) o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
    -/
    conv_lhs at h' => simp [HAdd.hAdd, Add.add]
    /-
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this : a.NF
      h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
      ⊢ Eq (HAdd.hAdd (e.oadd n a) o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
    -/
    have nf := ONote.add_nf a o
    /-
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this : a.NF
      h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
      nf : (HAdd.hAdd a o).NF
      ⊢ Eq (HAdd.hAdd (e.oadd n a) o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
    -/
    conv at nf => simp [HAdd.hAdd, Add.add]
    /-
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this : a.NF
      h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
      nf : (a.add o).NF
      ⊢ Eq (HAdd.hAdd (e.oadd n a) o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
    -/
    conv in _ + o => simp [HAdd.hAdd, Add.add]
    /-
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this : a.NF
      h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
      nf : (a.add o).NF
      ⊢ Eq ((e.oadd n a).add o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
    -/
    cases' h : add a o with e' n' a' <;>
      /-
        case zero
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NF
        h₂ : o.NF
        this : a.NF
        h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
        nf : (a.add o).NF
        h : Eq (a.add o) ONote.zero
        ⊢ Eq ((e.oadd n a).add o).repr (HAdd.hAdd (e.oadd n a).repr o.repr)
      -/
      /-
        🎉 no goals
      -/
      simp only [Add.add, add, addAux, h'.symm, h, add_assoc, repr] at nf h₁ ⊢
    /-
      case oadd
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this : a.NF
      h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
      e' : ONote
      n' : PNat
      a' : ONote
      h : Eq (a.add o) (e'.oadd n' a')
      nf : (e'.oadd n' a').NF
      ⊢ Eq (ONote.addAux.match_1 (fun x => ONote) (e.cmp e') (fun _ => e'.oadd n' a' …
    -/
    have := h₁.fst; haveI := nf.fst; have ee := cmp_compares e e'
    /-
      case oadd
      e : ONote
      n : PNat
      a o : ONote
      h₁ : (e.oadd n a).NF
      h₂ : o.NF
      this✝¹ : a.NF
      h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
      e' : ONote
      n' : PNat
      a' : ONote
      h : Eq (a.add o) (e'.oadd n' a')
      nf : (e'.oadd n' a').NF
      this✝ : e.NF
      this : e'.NF
      ee : (e.cmp e').Compares e e'
      ⊢ Eq (ONote.addAux.match_1 (fun x => ONote) (e.cmp e') (fun _ => e'.oadd n' a' …
    -/
    cases he : cmp e e' <;> simp only [he, Ordering.compares_gt, Ordering.compares_lt,
        Ordering.compares_eq, repr, gt_iff_lt, PNat.add_coe, Nat.cast_add] at ee ⊢
      /-
        case oadd.lt
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NF
        h₂ : o.NF
        this✝¹ : a.NF
        h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
        e' : ONote
        n' : PNat
        a' : ONote
        h : Eq (a.add o) (e'.oadd n' a')
        nf : (e'.oadd n' a').NF
        this✝ : e.NF
        this : e'.NF
        he : Eq (e.cmp e') Ordering.lt
        ee : LT.lt e e'
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e'.repr) ↑↑n') a'.repr) ( …
      -/
    · rw [← add_assoc, @add_absorp _ (repr e') (ω ^ repr e' * (n' : ℕ))]
        /-
          case oadd.lt.h₁
          e : ONote
          n : PNat
          a o : ONote
          h₁ : (e.oadd n a).NF
          h₂ : o.NF
          this✝¹ : a.NF
          h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
          e' : ONote
          n' : PNat
          a' : ONote
          h : Eq (a.add o) (e'.oadd n' a')
          nf : (e'.oadd n' a').NF
          this✝ : e.NF
          this : e'.NF
          he : Eq (e.cmp e') Ordering.lt
          ee : LT.lt e e'
          ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) (HPow.hPow Ordinal.o …
        -/
      · have := (h₁.below_of_lt ee).repr_lt
        /-
          case oadd.lt.h₁
          e : ONote
          n : PNat
          a o : ONote
          h₁ : (e.oadd n a).NF
          h₂ : o.NF
          this✝² : a.NF
          h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
          e' : ONote
          n' : PNat
          a' : ONote
          h : Eq (a.add o) (e'.oadd n' a')
          nf : (e'.oadd n' a').NF
          this✝¹ : e.NF
          this✝ : e'.NF
          he : Eq (e.cmp e') Ordering.lt
          ee : LT.lt e e'
          this : LT.lt (e.oadd n a).repr (HPow.hPow Ordinal.omega0 e'.repr)
          ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) (HPow.hPow Ordinal.o …
        -/
        unfold repr at this
        /-
          case oadd.lt.h₁
          e : ONote
          n : PNat
          a o : ONote
          h₁ : (e.oadd n a).NF
          h₂ : o.NF
          this✝² : a.NF
          h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
          e' : ONote
          n' : PNat
          a' : ONote
          h : Eq (a.add o) (e'.oadd n' a')
          nf : (e'.oadd n' a').NF
          this✝¹ : e.NF
          this✝ : e'.NF
          he : Eq (e.cmp e') Ordering.lt
          ee : LT.lt e e'
          this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
          ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) (HPow.hPow Ordinal.o …
        -/
        cases he' : e' <;> simp only [he', zero_def, opow_zero, repr, gt_iff_lt] at this ⊢ <;>
        /-
          case oadd.lt.h₁.zero
          e : ONote
          n : PNat
          a o : ONote
          h₁ : (e.oadd n a).NF
          h₂ : o.NF
          this✝² : a.NF
          h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
          e' : ONote
          n' : PNat
          a' : ONote
          h : Eq (a.add o) (e'.oadd n' a')
          nf : (e'.oadd n' a').NF
          this✝¹ : e.NF
          this✝ : e'.NF
          he : Eq (e.cmp e') Ordering.lt
          ee : LT.lt e e'
          he' : Eq e' ONote.zero
          this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
          ⊢ LT.lt (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) 1
        -/
        /-
          🎉 no goals
        -/
        exact lt_of_le_of_lt (le_add_right _ _) this
        /-
          🎉 no goals
        -/
      · simpa using (Ordinal.mul_le_mul_iff_left <| opow_pos (repr e') omega0_pos).2
          (Nat.cast_le.2 n'.pos)
      /-
        case oadd.eq
        e : ONote
        n : PNat
        a o : ONote
        h₁ : (e.oadd n a).NF
        h₂ : o.NF
        this✝¹ : a.NF
        h' : Eq (a.add o).repr (HAdd.hAdd a.repr o.repr)
        e' : ONote
        n' : PNat
        a' : ONote
        h : Eq (a.add o) (e'.oadd n' a')
        nf : (e'.oadd n' a').NF
        this✝ : e.NF
        this : e'.NF
        he : Eq (e.cmp e') Ordering.eq
        ee : Eq e e'
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd ↑↑n ↑↑ …
      -/
    · rw [ee, ← add_assoc, ← mul_add]
      /-
        🎉 no goals
      -/


theorem sub_nfBelow : ∀ {o₁ o₂ b}, NFBelow o₁ b → NF o₂ → NFBelow (o₁ - o₂) b
                         /-
                           o : ONote
                           b : Ordinal.{0}
                           x✝ : ONote.NFBelow 0 b
                           h₂ : o.NF
                           ⊢ (HSub.hSub 0 o).NFBelow b
                         -/
                                     /-
                                       🎉 no goals
                                     -/
  | 0, o, b, _, h₂ => by cases o <;> exact NFBelow.zero
                                     /-
                                       🎉 no goals
                                     -/
  | oadd _ _ _, 0, _, h₁, _ => h₁
  | oadd e₁ n₁ a₁, oadd e₂ n₂ a₂, b, h₁, h₂ => by
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      b : Ordinal.{0}
      h₁ : (e₁.oadd n₁ a₁).NFBelow b
      h₂ : (e₂.oadd n₂ a₂).NF
      ⊢ (HSub.hSub (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).NFBelow b
    -/
    have h' := sub_nfBelow h₁.snd h₂.snd
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      b : Ordinal.{0}
      h₁ : (e₁.oadd n₁ a₁).NFBelow b
      h₂ : (e₂.oadd n₂ a₂).NF
      h' : (HSub.hSub a₁ a₂).NFBelow e₁.repr
      ⊢ (HSub.hSub (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).NFBelow b
    -/
    simp only [HSub.hSub, Sub.sub, sub] at h' ⊢
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      b : Ordinal.{0}
      h₁ : (e₁.oadd n₁ a₁).NFBelow b
      h₂ : (e₂.oadd n₂ a₂).NF
      h' : (a₁.sub a₂).NFBelow e₁.repr
      ⊢ (ONote.sub.match_1 (fun x => ONote) (e₁.cmp e₂) (fun _ => 0) (fun _ => e₁.oa …
    -/
    have := @cmp_compares _ _ h₁.fst h₂.fst
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      b : Ordinal.{0}
      h₁ : (e₁.oadd n₁ a₁).NFBelow b
      h₂ : (e₂.oadd n₂ a₂).NF
      h' : (a₁.sub a₂).NFBelow e₁.repr
      this : (e₁.cmp e₂).Compares e₁ e₂
      ⊢ (ONote.sub.match_1 (fun x => ONote) (e₁.cmp e₂) (fun _ => 0) (fun _ => e₁.oa …
    -/
    cases h : cmp e₁ e₂
      /-
        case lt
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b
        h₂ : (e₂.oadd n₂ a₂).NF
        h' : (a₁.sub a₂).NFBelow e₁.repr
        this : (e₁.cmp e₂).Compares e₁ e₂
        h : Eq (e₁.cmp e₂) Ordering.lt
        ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.lt (fun _ => 0) (fun _ => e₁.oa …
      -/
    · apply NFBelow.zero
      /-
        🎉 no goals
      -/
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b
        h₂ : (e₂.oadd n₂ a₂).NF
        h' : (a₁.sub a₂).NFBelow e₁.repr
        this : (e₁.cmp e₂).Compares e₁ e₂
        h : Eq (e₁.cmp e₂) Ordering.eq
        ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁.oa …
      -/
    · rw [Nat.sub_eq]
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b
        h₂ : (e₂.oadd n₂ a₂).NF
        h' : (a₁.sub a₂).NFBelow e₁.repr
        this : (e₁.cmp e₂).Compares e₁ e₂
        h : Eq (e₁.cmp e₂) Ordering.eq
        ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁.oa …
      -/
      simp only [h, Ordering.compares_eq] at this
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b
        h₂ : (e₂.oadd n₂ a₂).NF
        h' : (a₁.sub a₂).NFBelow e₁.repr
        h : Eq (e₁.cmp e₂) Ordering.eq
        this : Eq e₁ e₂
        ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁.oa …
      -/
      subst e₂
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        n₂ : PNat
        a₂ : ONote
        b : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b
        h' : (a₁.sub a₂).NFBelow e₁.repr
        h₂ : (e₁.oadd n₂ a₂).NF
        h : Eq (e₁.cmp e₁) Ordering.eq
        ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁.oa …
      -/
      cases (n₁ : ℕ) - n₂
        /-
          case eq.zero
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          n₂ : PNat
          a₂ : ONote
          b : Ordinal.{0}
          h₁ : (e₁.oadd n₁ a₁).NFBelow b
          h' : (a₁.sub a₂).NFBelow e₁.repr
          h₂ : (e₁.oadd n₂ a₂).NF
          h : Eq (e₁.cmp e₁) Ordering.eq
          ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁.oa …
        -/
      · by_cases en : n₁ = n₂ <;> simp only [en, ↓reduceIte]
          /-
            case pos
            e₁ : ONote
            n₁ : PNat
            a₁ : ONote
            n₂ : PNat
            a₂ : ONote
            b : Ordinal.{0}
            h₁ : (e₁.oadd n₁ a₁).NFBelow b
            h' : (a₁.sub a₂).NFBelow e₁.repr
            h₂ : (e₁.oadd n₂ a₂).NF
            h : Eq (e₁.cmp e₁) Ordering.eq
            en : Eq n₁ n₂
            ⊢ (a₁.sub a₂).NFBelow b
          -/
        · exact h'.mono (le_of_lt h₁.lt)
          /-
            🎉 no goals
          -/
          /-
            case neg
            e₁ : ONote
            n₁ : PNat
            a₁ : ONote
            n₂ : PNat
            a₂ : ONote
            b : Ordinal.{0}
            h₁ : (e₁.oadd n₁ a₁).NFBelow b
            h' : (a₁.sub a₂).NFBelow e₁.repr
            h₂ : (e₁.oadd n₂ a₂).NF
            h : Eq (e₁.cmp e₁) Ordering.eq
            en : Not (Eq n₁ n₂)
            ⊢ ONote.NFBelow 0 b
          -/
        · exact NFBelow.zero
          /-
            🎉 no goals
          -/
        /-
          case eq.succ
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          n₂ : PNat
          a₂ : ONote
          b : Ordinal.{0}
          h₁ : (e₁.oadd n₁ a₁).NFBelow b
          h' : (a₁.sub a₂).NFBelow e₁.repr
          h₂ : (e₁.oadd n₂ a₂).NF
          h : Eq (e₁.cmp e₁) Ordering.eq
          n✝ : Nat
          ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁.oa …
        -/
      · exact NFBelow.oadd h₁.fst h₁.snd h₁.lt
        /-
          🎉 no goals
        -/
      /-
        case gt
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b
        h₂ : (e₂.oadd n₂ a₂).NF
        h' : (a₁.sub a₂).NFBelow e₁.repr
        this : (e₁.cmp e₂).Compares e₁ e₂
        h : Eq (e₁.cmp e₂) Ordering.gt
        ⊢ (ONote.sub.match_1 (fun x => ONote) Ordering.gt (fun _ => 0) (fun _ => e₁.oa …
      -/
    · exact h₁
      /-
        🎉 no goals
      -/


instance sub_nf (o₁ o₂) : ∀ [NF o₁] [NF o₂], NF (o₁ - o₂)
  | ⟨⟨b₁, h₁⟩⟩, h₂ => ⟨⟨b₁, sub_nfBelow h₁ h₂⟩⟩


@[simp]
theorem repr_sub : ∀ (o₁ o₂) [NF o₁] [NF o₂], repr (o₁ - o₂) = repr o₁ - repr o₂
                      /-
                        o : ONote
                        x✝ : ONote.NF 0
                        h₂ : o.NF
                        ⊢ Eq (HSub.hSub 0 o).repr (HSub.hSub (ONote.repr 0) o.repr)
                      -/
                                  /-
                                    🎉 no goals
                                  -/
  | 0, o, _, h₂ => by cases o <;> exact (Ordinal.zero_sub _).symm
                                  /-
                                    🎉 no goals
                                  -/
  | oadd _ _ _, 0, _, _ => (Ordinal.sub_zero _).symm
  | oadd e₁ n₁ a₁, oadd e₂ n₂ a₂, h₁, h₂ => by
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      ⊢ Eq (HSub.hSub (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HSub.hSub (e₁.oadd n₁ a …
    -/
    haveI := h₁.snd; haveI := h₂.snd; have h' := repr_sub a₁ a₂
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      this✝ : a₁.NF
      this : a₂.NF
      h' : Eq (HSub.hSub a₁ a₂).repr (HSub.hSub a₁.repr a₂.repr)
      ⊢ Eq (HSub.hSub (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HSub.hSub (e₁.oadd n₁ a …
    -/
    conv_lhs at h' => dsimp [HSub.hSub, Sub.sub, sub]
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      this✝ : a₁.NF
      this : a₂.NF
      h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
      ⊢ Eq (HSub.hSub (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HSub.hSub (e₁.oadd n₁ a …
    -/
    conv_lhs => dsimp only [HSub.hSub, Sub.sub]; dsimp only [sub]
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      this✝ : a₁.NF
      this : a₂.NF
      h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
      ⊢ Eq (ONote.sub.match_1 (fun x => ONote) (e₁.cmp e₂) (fun _ => 0) (fun _ => e₁ …
    -/
    have ee := @cmp_compares _ _ h₁.fst h₂.fst
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      this✝ : a₁.NF
      this : a₂.NF
      h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
      ee : (e₁.cmp e₂).Compares e₁ e₂
      ⊢ Eq (ONote.sub.match_1 (fun x => ONote) (e₁.cmp e₂) (fun _ => 0) (fun _ => e₁ …
    -/
    cases h : cmp e₁ e₂ <;> simp only [h] at ee
      /-
        case lt
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        this✝ : a₁.NF
        this : a₂.NF
        h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
        h : Eq (e₁.cmp e₂) Ordering.lt
        ee : Ordering.lt.Compares e₁ e₂
        ⊢ Eq (ONote.sub.match_1 (fun x => ONote) Ordering.lt (fun _ => 0) (fun _ => e₁ …
      -/
    · rw [Ordinal.sub_eq_zero_iff_le.2]
        /-
          case lt
          e₁ : ONote
          n₁ : PNat
          a₁ e₂ : ONote
          n₂ : PNat
          a₂ : ONote
          h₁ : (e₁.oadd n₁ a₁).NF
          h₂ : (e₂.oadd n₂ a₂).NF
          this✝ : a₁.NF
          this : a₂.NF
          h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
          h : Eq (e₁.cmp e₂) Ordering.lt
          ee : Ordering.lt.Compares e₁ e₂
          ⊢ Eq (ONote.sub.match_1 (fun x => ONote) Ordering.lt (fun _ => 0) (fun _ => e₁ …
        -/
      · rfl
        /-
          🎉 no goals
        -/
      /-
        case lt
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        this✝ : a₁.NF
        this : a₂.NF
        h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
        h : Eq (e₁.cmp e₂) Ordering.lt
        ee : Ordering.lt.Compares e₁ e₂
        ⊢ LE.le (e₁.oadd n₁ a₁).repr (e₂.oadd n₂ a₂).repr
      -/
      exact le_of_lt (oadd_lt_oadd_1 h₁ ee)
      /-
        🎉 no goals
      -/
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        this✝ : a₁.NF
        this : a₂.NF
        h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
        h : Eq (e₁.cmp e₂) Ordering.eq
        ee : Ordering.eq.Compares e₁ e₂
        ⊢ Eq (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁ …
      -/
    · change e₁ = e₂ at ee
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        this✝ : a₁.NF
        this : a₂.NF
        h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
        h : Eq (e₁.cmp e₂) Ordering.eq
        ee : Eq e₁ e₂
        ⊢ Eq (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁ …
      -/
      subst e₂
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        this✝ : a₁.NF
        this : a₂.NF
        h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
        h₂ : (e₁.oadd n₂ a₂).NF
        h : Eq (e₁.cmp e₁) Ordering.eq
        ⊢ Eq (ONote.sub.match_1 (fun x => ONote) Ordering.eq (fun _ => 0) (fun _ => e₁ …
      -/
      dsimp only
      /-
        case eq
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        this✝ : a₁.NF
        this : a₂.NF
        h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
        h₂ : (e₁.oadd n₂ a₂).NF
        h : Eq (e₁.cmp e₁) Ordering.eq
        ⊢ Eq (ONote.ofNat.match_1 (fun x => ONote) (HSub.hSub ↑n₁ ↑n₂) (fun _ => ite ( …
      -/
      cases mn : (n₁ : ℕ) - n₂ <;> dsimp only
        /-
          case eq.zero
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          n₂ : PNat
          a₂ : ONote
          h₁ : (e₁.oadd n₁ a₁).NF
          this✝ : a₁.NF
          this : a₂.NF
          h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
          h₂ : (e₁.oadd n₂ a₂).NF
          h : Eq (e₁.cmp e₁) Ordering.eq
          mn : Eq (HSub.hSub ↑n₁ ↑n₂) 0
          ⊢ Eq (ite (Eq n₁ n₂) (a₁.sub a₂) 0).repr (HSub.hSub (e₁.oadd n₁ a₁).repr (e₁.o …
        -/
      · by_cases en : n₁ = n₂
          /-
            case pos
            e₁ : ONote
            n₁ : PNat
            a₁ : ONote
            n₂ : PNat
            a₂ : ONote
            h₁ : (e₁.oadd n₁ a₁).NF
            this✝ : a₁.NF
            this : a₂.NF
            h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
            h₂ : (e₁.oadd n₂ a₂).NF
            h : Eq (e₁.cmp e₁) Ordering.eq
            mn : Eq (HSub.hSub ↑n₁ ↑n₂) 0
            en : Eq n₁ n₂
            ⊢ Eq (ite (Eq n₁ n₂) (a₁.sub a₂) 0).repr (HSub.hSub (e₁.oadd n₁ a₁).repr (e₁.o …
          -/
        · simpa [en]
          /-
            🎉 no goals
          -/
          /-
            case neg
            e₁ : ONote
            n₁ : PNat
            a₁ : ONote
            n₂ : PNat
            a₂ : ONote
            h₁ : (e₁.oadd n₁ a₁).NF
            this✝ : a₁.NF
            this : a₂.NF
            h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
            h₂ : (e₁.oadd n₂ a₂).NF
            h : Eq (e₁.cmp e₁) Ordering.eq
            mn : Eq (HSub.hSub ↑n₁ ↑n₂) 0
            en : Not (Eq n₁ n₂)
            ⊢ Eq (ite (Eq n₁ n₂) (a₁.sub a₂) 0).repr (HSub.hSub (e₁.oadd n₁ a₁).repr (e₁.o …
          -/
        · simp only [en, ite_false]
          exact
            (Ordinal.sub_eq_zero_iff_le.2 <|
                le_of_lt <|
                  oadd_lt_oadd_2 h₁ <|
                    lt_of_le_of_ne (tsub_eq_zero_iff_le.1 mn) (mt PNat.eq en)).symm
        /-
          case eq.succ
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          n₂ : PNat
          a₂ : ONote
          h₁ : (e₁.oadd n₁ a₁).NF
          this✝ : a₁.NF
          this : a₂.NF
          h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
          h₂ : (e₁.oadd n₂ a₂).NF
          h : Eq (e₁.cmp e₁) Ordering.eq
          n✝ : Nat
          mn : Eq (HSub.hSub ↑n₁ ↑n₂) (HAdd.hAdd n✝ 1)
          ⊢ Eq (e₁.oadd n✝.succPNat a₁).repr (HSub.hSub (e₁.oadd n₁ a₁).repr (e₁.oadd n₂ …
        -/
      · simp [Nat.succPNat]
        rw [(tsub_eq_iff_eq_add_of_le <| le_of_lt <| Nat.lt_of_sub_eq_succ mn).1 mn, add_comm,
          Nat.cast_add, mul_add, add_assoc, add_sub_add_cancel]
        refine
          (Ordinal.sub_eq_of_add_eq <|
              add_absorp h₂.snd'.repr_lt <| le_trans ?_ (le_add_right _ _)).symm
        /-
          case eq.succ
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          n₂ : PNat
          a₂ : ONote
          h₁ : (e₁.oadd n₁ a₁).NF
          this✝ : a₁.NF
          this : a₂.NF
          h' : Eq (a₁.sub a₂).repr (HSub.hSub a₁.repr a₂.repr)
          h₂ : (e₁.oadd n₂ a₂).NF
          h : Eq (e₁.cmp e₁) Ordering.eq
          n✝ : Nat
          mn : Eq (HSub.hSub ↑n₁ ↑n₂) (HAdd.hAdd n✝ 1)
          ⊢ LE.le (HPow.hPow Ordinal.omega0 e₁.repr) (HMul.hMul (HPow.hPow Ordinal.omega …
        -/
        exact Ordinal.le_mul_left _ (Nat.cast_lt.2 <| Nat.succ_pos _)
        /-
          🎉 no goals
        -/
    · exact
        (Ordinal.sub_eq_of_add_eq <|
            add_absorp (h₂.below_of_lt ee).repr_lt <| omega0_le_oadd _ _ _).symm


/-- Multiplication of ordinal notations (correct only for normal input) -/
def mul : ONote → ONote → ONote
  | 0, _ => 0
  | _, 0 => 0
  | o₁@(oadd e₁ n₁ a₁), oadd e₂ n₂ a₂ =>
    if e₂ = 0 then oadd e₁ (n₁ * n₂) a₁ else oadd (e₁ + e₂) n₂ (mul o₁ a₂)


instance : Mul ONote :=
  ⟨mul⟩


instance : MulZeroClass ONote where
  mul := (· * ·)
  zero := 0
                   /-
                     o : ONote
                     ⊢ Eq (HMul.hMul 0 o) 0
                   -/
                               /-
                                 🎉 no goals
                               -/
  zero_mul o := by cases o <;> rfl
                               /-
                                 🎉 no goals
                               -/
                   /-
                     o : ONote
                     ⊢ Eq (HMul.hMul o 0) 0
                   -/
                               /-
                                 🎉 no goals
                               -/
  mul_zero o := by cases o <;> rfl
                               /-
                                 🎉 no goals
                               -/


theorem oadd_mul (e₁ n₁ a₁ e₂ n₂ a₂) :
    oadd e₁ n₁ a₁ * oadd e₂ n₂ a₂ =
      if e₂ = 0 then oadd e₁ (n₁ * n₂) a₁ else oadd (e₁ + e₂) n₂ (oadd e₁ n₁ a₁ * a₂) :=
  rfl


theorem oadd_mul_nfBelow {e₁ n₁ a₁ b₁} (h₁ : NFBelow (oadd e₁ n₁ a₁) b₁) :
    ∀ {o₂ b₂}, NFBelow o₂ b₂ → NFBelow (oadd e₁ n₁ a₁ * o₂) (repr e₁ + b₂)
  | 0, _, _ => NFBelow.zero
  | oadd e₂ n₂ a₂, b₂, h₂ => by
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      b₁ : Ordinal.{0}
      h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
      e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      b₂ : Ordinal.{0}
      h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
      ⊢ (HMul.hMul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).NFBelow (HAdd.hAdd e₁.repr b₂)
    -/
    have IH := oadd_mul_nfBelow h₁ h₂.snd
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ : ONote
      b₁ : Ordinal.{0}
      h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
      e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      b₂ : Ordinal.{0}
      h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
      IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
      ⊢ (HMul.hMul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).NFBelow (HAdd.hAdd e₁.repr b₂)
    -/
    by_cases e0 : e₂ = 0 <;> simp only [e0, oadd_mul, ↓reduceIte]
      /-
        case pos
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        b₁ : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
        e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b₂ : Ordinal.{0}
        h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
        IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
        e0 : Eq e₂ 0
        ⊢ (e₁.oadd (HMul.hMul n₁ n₂) a₁).NFBelow (HAdd.hAdd e₁.repr b₂)
      -/
    · apply NFBelow.oadd h₁.fst h₁.snd
      /-
        case pos
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        b₁ : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
        e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b₂ : Ordinal.{0}
        h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
        IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
        e0 : Eq e₂ 0
        ⊢ LT.lt e₁.repr (HAdd.hAdd e₁.repr b₂)
      -/
      simpa using (add_lt_add_iff_left (repr e₁)).2 (lt_of_le_of_lt (Ordinal.zero_le _) h₂.lt)
      /-
        🎉 no goals
      -/
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        b₁ : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
        e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b₂ : Ordinal.{0}
        h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
        IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
        e0 : Not (Eq e₂ 0)
        ⊢ ((HAdd.hAdd e₁ e₂).oadd n₂ (HMul.hMul (e₁.oadd n₁ a₁) a₂)).NFBelow (HAdd.hAd …
      -/
    · haveI := h₁.fst
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        b₁ : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
        e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b₂ : Ordinal.{0}
        h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
        IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
        e0 : Not (Eq e₂ 0)
        this : e₁.NF
        ⊢ ((HAdd.hAdd e₁ e₂).oadd n₂ (HMul.hMul (e₁.oadd n₁ a₁) a₂)).NFBelow (HAdd.hAd …
      -/
      haveI := h₂.fst
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ : ONote
        b₁ : Ordinal.{0}
        h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
        e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        b₂ : Ordinal.{0}
        h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
        IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
        e0 : Not (Eq e₂ 0)
        this✝ : e₁.NF
        this : e₂.NF
        ⊢ ((HAdd.hAdd e₁ e₂).oadd n₂ (HMul.hMul (e₁.oadd n₁ a₁) a₂)).NFBelow (HAdd.hAd …
      -/
      apply NFBelow.oadd
        /-
          case neg.a
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          b₁ : Ordinal.{0}
          h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
          e₂ : ONote
          n₂ : PNat
          a₂ : ONote
          b₂ : Ordinal.{0}
          h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
          IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
          e0 : Not (Eq e₂ 0)
          this✝ : e₁.NF
          this : e₂.NF
          ⊢ (HAdd.hAdd e₁ e₂).NF
        -/
      · infer_instance
        /-
          🎉 no goals
        -/
        /-
          case neg.a
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          b₁ : Ordinal.{0}
          h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
          e₂ : ONote
          n₂ : PNat
          a₂ : ONote
          b₂ : Ordinal.{0}
          h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
          IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
          e0 : Not (Eq e₂ 0)
          this✝ : e₁.NF
          this : e₂.NF
          ⊢ (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁ e₂).repr
        -/
      · rwa [repr_add]
        /-
          🎉 no goals
        -/
        /-
          case neg.a
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          b₁ : Ordinal.{0}
          h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
          e₂ : ONote
          n₂ : PNat
          a₂ : ONote
          b₂ : Ordinal.{0}
          h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
          IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
          e0 : Not (Eq e₂ 0)
          this✝ : e₁.NF
          this : e₂.NF
          ⊢ LT.lt (HAdd.hAdd e₁ e₂).repr (HAdd.hAdd e₁.repr b₂)
        -/
      · rw [repr_add, add_lt_add_iff_left]
        /-
          case neg.a
          e₁ : ONote
          n₁ : PNat
          a₁ : ONote
          b₁ : Ordinal.{0}
          h₁ : (e₁.oadd n₁ a₁).NFBelow b₁
          e₂ : ONote
          n₂ : PNat
          a₂ : ONote
          b₂ : Ordinal.{0}
          h₂ : (e₂.oadd n₂ a₂).NFBelow b₂
          IH : (HMul.hMul (e₁.oadd n₁ a₁) a₂).NFBelow (HAdd.hAdd e₁.repr e₂.repr)
          e0 : Not (Eq e₂ 0)
          this✝ : e₁.NF
          this : e₂.NF
          ⊢ LT.lt e₂.repr b₂
        -/
        exact h₂.lt
        /-
          🎉 no goals
        -/


instance mul_nf : ∀ (o₁ o₂) [NF o₁] [NF o₂], NF (o₁ * o₂)
                      /-
                        o : ONote
                        x✝ : ONote.NF 0
                        h₂ : o.NF
                        ⊢ (HMul.hMul 0 o).NF
                      -/
                                  /-
                                    🎉 no goals
                                  -/
  | 0, o, _, h₂ => by cases o <;> exact NF.zero
                                  /-
                                    🎉 no goals
                                  -/
  | oadd _ _ _, _, ⟨⟨_, hb₁⟩⟩, ⟨⟨_, hb₂⟩⟩ => ⟨⟨_, oadd_mul_nfBelow hb₁ hb₂⟩⟩


@[simp]
theorem repr_mul : ∀ (o₁ o₂) [NF o₁] [NF o₂], repr (o₁ * o₂) = repr o₁ * repr o₂
                      /-
                        o : ONote
                        x✝ : ONote.NF 0
                        h₂ : o.NF
                        ⊢ Eq (HMul.hMul 0 o).repr (HMul.hMul (ONote.repr 0) o.repr)
                      -/
                                  /-
                                    🎉 no goals
                                  -/
  | 0, o, _, h₂ => by cases o <;> exact (zero_mul _).symm
                                  /-
                                    🎉 no goals
                                  -/
  | oadd _ _ _, 0, _, _ => (mul_zero _).symm
  | oadd e₁ n₁ a₁, oadd e₂ n₂ a₂, h₁, h₂ => by
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      ⊢ Eq (HMul.hMul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (e₁.oadd n₁ a …
    -/
    have IH : repr (mul _ _) = _ := @repr_mul _ _ h₁ h₂.snd
    conv =>
      lhs
      simp [(· * ·)]
    have ao : repr a₁ + ω ^ repr e₁ * (n₁ : ℕ) = ω ^ repr e₁ * (n₁ : ℕ) := by
      apply add_absorp h₁.snd'.repr_lt
      simpa using (Ordinal.mul_le_mul_iff_left <| opow_pos _ omega0_pos).2 (Nat.cast_le.2 n₁.2)
    /-
      e₁ : ONote
      n₁ : PNat
      a₁ e₂ : ONote
      n₂ : PNat
      a₂ : ONote
      h₁ : (e₁.oadd n₁ a₁).NF
      h₂ : (e₂.oadd n₂ a₂).NF
      IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
      ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
      ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (e₁.oadd n₁ a₁) …
    -/
    by_cases e0 : e₂ = 0
      /-
        case pos
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Eq e₂ 0
        ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (e₁.oadd n₁ a₁) …
      -/
    · cases' Nat.exists_eq_succ_of_ne_zero n₂.ne_zero with x xe
      /-
        case pos.intro
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Eq e₂ 0
        x : Nat
        xe : Eq (↑n₂) x.succ
        ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (e₁.oadd n₁ a₁) …
      -/
      simp only [e0, repr, PNat.mul_coe, natCast_mul, opow_zero, one_mul]
      /-
        case pos.intro
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Eq e₂ 0
        x : Nat
        xe : Eq (↑n₂) x.succ
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) (HMul.hMul ↑↑n₁  …
      -/
      simp only [xe, h₂.zero_of_zero e0, repr, add_zero]
      /-
        case pos.intro
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Eq e₂ 0
        x : Nat
        xe : Eq (↑n₂) x.succ
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) (HMul.hMul ↑↑n₁  …
      -/
      rw [natCast_succ x, add_mul_succ _ ao, mul_assoc]
      /-
        🎉 no goals
      -/
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (e₁.oadd n₁ a₁) …
      -/
    · simp only [repr]
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (HAdd.hAdd (HMu …
      -/
      haveI := h₁.fst
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        this : e₁.NF
        ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (HAdd.hAdd (HMu …
      -/
      haveI := h₂.fst
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        this✝ : e₁.NF
        this : e₂.NF
        ⊢ Eq (Mul.mul (e₁.oadd n₁ a₁) (e₂.oadd n₂ a₂)).repr (HMul.hMul (HAdd.hAdd (HMu …
      -/
      simp only [Mul.mul, mul, e0, ite_false, repr.eq_2, repr_add, opow_add, IH, repr, mul_add]
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        this✝ : e₁.NF
        this : e₂.NF
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) (HPow …
      -/
      rw [← mul_assoc]
      /-
        case neg
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        this✝ : e₁.NF
        this : e₂.NF
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) (HPow …
      -/
      congr 2
      /-
        case neg.e_a.e_a
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        this✝ : e₁.NF
        this : e₂.NF
        ⊢ Eq (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) (HPow.hPow Ordinal.omega0 e …
      -/
      have := mt repr_inj.1 e0
      rw [add_mul_limit ao (isLimit_opow_left isLimit_omega0 this), mul_assoc,
        mul_omega0_dvd (Nat.cast_pos'.2 n₁.pos) (nat_lt_omega0 _)]
      /-
        case neg.e_a.e_a
        e₁ : ONote
        n₁ : PNat
        a₁ e₂ : ONote
        n₂ : PNat
        a₂ : ONote
        h₁ : (e₁.oadd n₁ a₁).NF
        h₂ : (e₂.oadd n₂ a₂).NF
        IH : Eq ((e₁.oadd n₁ a₁).mul a₂).repr (HMul.hMul (e₁.oadd n₁ a₁).repr a₂.repr)
        ao : Eq (HAdd.hAdd a₁.repr (HMul.hMul (HPow.hPow Ordinal.omega0 e₁.repr) ↑↑n₁) …
        e0 : Not (Eq e₂ 0)
        this✝¹ : e₁.NF
        this✝ : e₂.NF
        this : Not (Eq e₂.repr (ONote.repr 0))
        ⊢ Dvd.dvd Ordinal.omega0 (HPow.hPow Ordinal.omega0 e₂.repr)
      -/
      simpa using opow_dvd_opow ω (one_le_iff_ne_zero.2 this)
      /-
        🎉 no goals
      -/


/-- Calculate division and remainder of `o` mod `ω`:

`split' o = (a, n)` means `o = ω * a + n`. -/
def split' : ONote → ONote × ℕ
  | 0 => (0, 0)
  | oadd e n a =>
    if e = 0 then (0, n)
    else
      let (a', m) := split' a
      (oadd (e - 1) n a', m)


/-- Calculate division and remainder of `o` mod `ω`:

`split o = (a, n)` means `o = a + n`, where `ω ∣ a`. -/
def split : ONote → ONote × ℕ
  | 0 => (0, 0)
  | oadd e n a =>
    if e = 0 then (0, n)
    else
      let (a', m) := split a
      (oadd e n a', m)


/-- `scale x o` is the ordinal notation for `ω ^ x * o`. -/
def scale (x : ONote) : ONote → ONote
  | 0 => 0
  | oadd e n a => oadd (x + e) n (scale x a)


/-- `mulNat o n` is the ordinal notation for `o * n`. -/
def mulNat : ONote → ℕ → ONote
  | 0, _ => 0
  | _, 0 => 0
  | oadd e n a, m + 1 => oadd e (n * m.succPNat) a


/-- Auxiliary definition to compute the ordinal notation for the ordinal exponentiation in `opow` -/
def opowAux (e a0 a : ONote) : ℕ → ℕ → ONote
  | _, 0 => 0
  | 0, m + 1 => oadd e m.succPNat 0
  | k + 1, m => scale (e + mulNat a0 k) a + (opowAux e a0 a k m)


/-- Auxiliary definition to compute the ordinal notation for the ordinal exponentiation in `opow` -/
def opowAux2 (o₂ : ONote) (o₁ : ONote × ℕ) : ONote :=
  match o₁ with
  | (0, 0) => if o₂ = 0 then 1 else 0
  | (0, 1) => 1
  | (0, m + 1) =>
    let (b', k) := split' o₂
    oadd b' (m.succPNat ^ k) 0
  | (a@(oadd a0 _ _), m) =>
    match split o₂ with
    | (b, 0) => oadd (a0 * b) 1 0
    | (b, k + 1) =>
      let eb := a0 * b
      scale (eb + mulNat a0 k) a + opowAux eb a0 (mulNat a m) k m


/-- `opow o₁ o₂` calculates the ordinal notation for the ordinal exponential `o₁ ^ o₂`. -/
def opow (o₁ o₂ : ONote) : ONote := opowAux2 o₂ (split o₁)


instance : Pow ONote ONote :=
  ⟨opow⟩


theorem opow_def (o₁ o₂ : ONote) : o₁ ^ o₂ = opowAux2 o₂ (split o₁) :=
  rfl


theorem split_eq_scale_split' : ∀ {o o' m} [NF o], split' o = (o', m) → split o = (scale 1 o', m)
                         /-
                           o' : ONote
                           m : Nat
                           x✝ : ONote.NF 0
                           p : Eq (ONote.split' 0) { fst := o', snd := m }
                           ⊢ Eq (ONote.split 0) { fst := ONote.scale 1 o', snd := m }
                         -/
  | 0, o', m, _, p => by injection p; substs o' m; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/
  | oadd e n a, o', m, h, p => by
    /-
      e : ONote
      n : PNat
      a o' : ONote
      m : Nat
      h : (e.oadd n a).NF
      p : Eq (e.oadd n a).split' { fst := o', snd := m }
      ⊢ Eq (e.oadd n a).split { fst := ONote.scale 1 o', snd := m }
    -/
    by_cases e0 : e = 0 <;> simp only [split', e0, ↓reduceIte, Prod.mk.injEq, split] at p ⊢
      /-
        case pos
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Eq e 0
        p : And (Eq 0 o') (Eq (↑n) m)
        ⊢ And (Eq 0 (ONote.scale 1 o')) (Eq (↑n) m)
      -/
    · rcases p with ⟨rfl, rfl⟩
      /-
        case pos.intro
        e : ONote
        n : PNat
        a : ONote
        h : (e.oadd n a).NF
        e0 : Eq e 0
        ⊢ And (Eq 0 (ONote.scale 1 0)) (Eq ↑n ↑n)
      -/
      exact ⟨rfl, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        p : And (Eq ((HSub.hSub e 1).oadd n a.split'.1) o') (Eq a.split'.2 m)
        ⊢ And (Eq (e.oadd n a.split.1) (ONote.scale 1 o')) (Eq a.split.2 m)
      -/
    · revert p
      /-
        case neg
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        ⊢ And (Eq ((HSub.hSub e 1).oadd n a.split'.1) o') (Eq a.split'.2 m) → And (Eq  …
      -/
      cases' h' : split' a with a' m'
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      haveI := h.fst
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this : e.NF
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      haveI := h.snd
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝ : e.NF
        this : a.NF
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      simp only [split_eq_scale_split' h', and_imp]
      have : 1 + (e - 1) = e := by
        refine repr_inj.1 ?_
        simp only [repr_add, repr, opow_zero, Nat.succPNat_coe, Nat.cast_one, mul_one, add_zero,
          repr_sub]
        have := mt repr_inj.1 e0
        exact Ordinal.add_sub_cancel_of_le <| one_le_iff_ne_zero.2 this
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝¹ : e.NF
        this✝ : a.NF
        this : Eq (HAdd.hAdd 1 (HSub.hSub e 1)) e
        ⊢ Eq ((HSub.hSub e 1).oadd n a') o' → Eq m' m → And (Eq (e.oadd n (ONote.scale …
      -/
      intros
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝¹ : e.NF
        this✝ : a.NF
        this : Eq (HAdd.hAdd 1 (HSub.hSub e 1)) e
        a✝¹ : Eq ((HSub.hSub e 1).oadd n a') o'
        a✝ : Eq m' m
        ⊢ And (Eq (e.oadd n (ONote.scale 1 a')) (ONote.scale 1 o')) (Eq m' m)
      -/
      substs o' m
      /-
        case neg.mk
        e : ONote
        n : PNat
        a : ONote
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝¹ : e.NF
        this✝ : a.NF
        this : Eq (HAdd.hAdd 1 (HSub.hSub e 1)) e
        ⊢ And (Eq (e.oadd n (ONote.scale 1 a')) (ONote.scale 1 ((HSub.hSub e 1).oadd n …
      -/
      simp [scale, this]
      /-
        🎉 no goals
      -/


theorem nf_repr_split' : ∀ {o o' m} [NF o], split' o = (o', m) → NF o' ∧ repr o = ω * repr o' + m
                         /-
                           o' : ONote
                           m : Nat
                           x✝ : ONote.NF 0
                           p : Eq (ONote.split' 0) { fst := o', snd := m }
                           ⊢ And o'.NF (Eq (ONote.repr 0) (HAdd.hAdd (HMul.hMul Ordinal.omega0 o'.repr) ↑ …
                         -/
  | 0, o', m, _, p => by injection p; substs o' m; simp [NF.zero]
                                                   /-
                                                     🎉 no goals
                                                   -/
  | oadd e n a, o', m, h, p => by
    /-
      e : ONote
      n : PNat
      a o' : ONote
      m : Nat
      h : (e.oadd n a).NF
      p : Eq (e.oadd n a).split' { fst := o', snd := m }
      ⊢ And o'.NF (Eq (e.oadd n a).repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 o'.repr …
    -/
    by_cases e0 : e = 0 <;> simp [e0, split, split'] at p ⊢
      /-
        case pos
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Eq e 0
        p : And (Eq 0 o') (Eq (↑n) m)
        ⊢ And o'.NF (Eq (HAdd.hAdd (↑↑n) a.repr) (HAdd.hAdd (HMul.hMul Ordinal.omega0  …
      -/
    · rcases p with ⟨rfl, rfl⟩
      /-
        case pos.intro
        e : ONote
        n : PNat
        a : ONote
        h : (e.oadd n a).NF
        e0 : Eq e 0
        ⊢ And (ONote.NF 0) (Eq (HAdd.hAdd (↑↑n) a.repr) (HAdd.hAdd (HMul.hMul Ordinal. …
      -/
      simp [h.zero_of_zero e0, NF.zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        p : And (Eq ((HSub.hSub e 1).oadd n a.split'.1) o') (Eq a.split'.2 m)
        ⊢ And o'.NF (Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a …
      -/
    · revert p
      /-
        case neg
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        ⊢ And (Eq ((HSub.hSub e 1).oadd n a.split'.1) o') (Eq a.split'.2 m) → And o'.N …
      -/
      cases' h' : split' a with a' m'
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      haveI := h.fst
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this : e.NF
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      haveI := h.snd
      /-
        case neg.mk
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝ : e.NF
        this : a.NF
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      cases' nf_repr_split' h' with IH₁ IH₂
      /-
        case neg.mk.intro
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝ : e.NF
        this : a.NF
        IH₁ : a'.NF
        IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
        ⊢ And (Eq ((HSub.hSub e 1).oadd n { fst := a', snd := m' }.1) o') (Eq { fst := …
      -/
      simp only [IH₂, and_imp]
      /-
        case neg.mk.intro
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝ : e.NF
        this : a.NF
        IH₁ : a'.NF
        IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
        ⊢ Eq ((HSub.hSub e 1).oadd n a') o' → Eq m' m → And o'.NF (Eq (HAdd.hAdd (HMul …
      -/
      intros
      /-
        case neg.mk.intro
        e : ONote
        n : PNat
        a o' : ONote
        m : Nat
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝ : e.NF
        this : a.NF
        IH₁ : a'.NF
        IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
        a✝¹ : Eq ((HSub.hSub e 1).oadd n a') o'
        a✝ : Eq m' m
        ⊢ And o'.NF (Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) ( …
      -/
      substs o' m
      have : (ω : Ordinal.{0}) ^ repr e = ω ^ (1 : Ordinal.{0}) * ω ^ (repr e - 1) := by
        have := mt repr_inj.1 e0
        rw [← opow_add, Ordinal.add_sub_cancel_of_le (one_le_iff_ne_zero.2 this)]
      /-
        case neg.mk.intro
        e : ONote
        n : PNat
        a : ONote
        h : (e.oadd n a).NF
        e0 : Not (Eq e 0)
        a' : ONote
        m' : Nat
        h' : Eq a.split' { fst := a', snd := m' }
        this✝¹ : e.NF
        this✝ : a.NF
        IH₁ : a'.NF
        IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
        this : Eq (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul (HPow.hPow Ordinal.omeg …
        ⊢ And ((HSub.hSub e 1).oadd n a').NF (Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordi …
      -/
      refine ⟨NF.oadd (by infer_instance) _ ?_, ?_⟩
        /-
          case neg.mk.intro.refine_1
          e : ONote
          n : PNat
          a : ONote
          h : (e.oadd n a).NF
          e0 : Not (Eq e 0)
          a' : ONote
          m' : Nat
          h' : Eq a.split' { fst := a', snd := m' }
          this✝¹ : e.NF
          this✝ : a.NF
          IH₁ : a'.NF
          IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
          this : Eq (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul (HPow.hPow Ordinal.omeg …
          ⊢ a'.NFBelow (HSub.hSub e 1).repr
        -/
      · simp at this ⊢
        refine
          IH₁.below_of_lt'
            ((Ordinal.mul_lt_mul_iff_left omega0_pos).1 <| lt_of_le_of_lt (le_add_right _ m') ?_)
        /-
          case neg.mk.intro.refine_1
          e : ONote
          n : PNat
          a : ONote
          h : (e.oadd n a).NF
          e0 : Not (Eq e 0)
          a' : ONote
          m' : Nat
          h' : Eq a.split' { fst := a', snd := m' }
          this✝¹ : e.NF
          this✝ : a.NF
          IH₁ : a'.NF
          IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
          this : Eq (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul Ordinal.omega0 (HPow.hP …
          ⊢ LT.lt (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m') (HMul.hMul Ordinal. …
        -/
        rw [← this, ← IH₂]
        /-
          case neg.mk.intro.refine_1
          e : ONote
          n : PNat
          a : ONote
          h : (e.oadd n a).NF
          e0 : Not (Eq e 0)
          a' : ONote
          m' : Nat
          h' : Eq a.split' { fst := a', snd := m' }
          this✝¹ : e.NF
          this✝ : a.NF
          IH₁ : a'.NF
          IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
          this : Eq (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul Ordinal.omega0 (HPow.hP …
          ⊢ LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
        -/
        exact h.snd'.repr_lt
        /-
          🎉 no goals
        -/
        /-
          case neg.mk.intro.refine_2
          e : ONote
          n : PNat
          a : ONote
          h : (e.oadd n a).NF
          e0 : Not (Eq e 0)
          a' : ONote
          m' : Nat
          h' : Eq a.split' { fst := a', snd := m' }
          this✝¹ : e.NF
          this✝ : a.NF
          IH₁ : a'.NF
          IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
          this : Eq (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul (HPow.hPow Ordinal.omeg …
          ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) (HAdd.hAdd ( …
        -/
      · rw [this]
        /-
          case neg.mk.intro.refine_2
          e : ONote
          n : PNat
          a : ONote
          h : (e.oadd n a).NF
          e0 : Not (Eq e 0)
          a' : ONote
          m' : Nat
          h' : Eq a.split' { fst := a', snd := m' }
          this✝¹ : e.NF
          this✝ : a.NF
          IH₁ : a'.NF
          IH₂ : Eq a.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a'.repr) ↑m')
          this : Eq (HPow.hPow Ordinal.omega0 e.repr) (HMul.hMul (HPow.hPow Ordinal.omeg …
          ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow Ordinal.omega0 1) (HPow.hPow  …
        -/
        simp [mul_add, mul_assoc, add_assoc]
        /-
          🎉 no goals
        -/


theorem scale_eq_mul (x) [NF x] : ∀ (o) [NF o], scale x o = oadd x 1 0 * o
  | 0, _ => rfl
  | oadd e n a, h => by
    /-
      x : ONote
      inst✝ : x.NF
      e : ONote
      n : PNat
      a : ONote
      h : (e.oadd n a).NF
      ⊢ Eq (x.scale (e.oadd n a)) (HMul.hMul (x.oadd 1 0) (e.oadd n a))
    -/
    simp only [HMul.hMul]; simp only [scale]
    /-
      x : ONote
      inst✝ : x.NF
      e : ONote
      n : PNat
      a : ONote
      h : (e.oadd n a).NF
      ⊢ Eq ((HAdd.hAdd x e).oadd n (x.scale a)) (Mul.mul (x.oadd 1 0) (e.oadd n a))
    -/
    haveI := h.snd
    /-
      x : ONote
      inst✝ : x.NF
      e : ONote
      n : PNat
      a : ONote
      h : (e.oadd n a).NF
      this : a.NF
      ⊢ Eq ((HAdd.hAdd x e).oadd n (x.scale a)) (Mul.mul (x.oadd 1 0) (e.oadd n a))
    -/
    by_cases e0 : e = 0
      /-
        case pos
        x : ONote
        inst✝ : x.NF
        e : ONote
        n : PNat
        a : ONote
        h : (e.oadd n a).NF
        this : a.NF
        e0 : Eq e 0
        ⊢ Eq ((HAdd.hAdd x e).oadd n (x.scale a)) (Mul.mul (x.oadd 1 0) (e.oadd n a))
      -/
    · simp_rw [scale_eq_mul]
      simp [Mul.mul, mul, scale_eq_mul, e0, h.zero_of_zero,
        show x + 0 = x from repr_inj.1 (by simp)]
      /-
        case neg
        x : ONote
        inst✝ : x.NF
        e : ONote
        n : PNat
        a : ONote
        h : (e.oadd n a).NF
        this : a.NF
        e0 : Not (Eq e 0)
        ⊢ Eq ((HAdd.hAdd x e).oadd n (x.scale a)) (Mul.mul (x.oadd 1 0) (e.oadd n a))
      -/
    · simp [e0, Mul.mul, mul, scale_eq_mul, (· * ·)]
      /-
        🎉 no goals
      -/


instance nf_scale (x) [NF x] (o) [NF o] : NF (scale x o) := by
  /-
    x : ONote
    inst✝¹ : x.NF
    o : ONote
    inst✝ : o.NF
    ⊢ (x.scale o).NF
  -/
  rw [scale_eq_mul]
  /-
    x : ONote
    inst✝¹ : x.NF
    o : ONote
    inst✝ : o.NF
    ⊢ (HMul.hMul (x.oadd 1 0) o).NF
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem repr_scale (x) [NF x] (o) [NF o] : repr (scale x o) = ω ^ repr x * repr o := by
  /-
    x : ONote
    inst✝¹ : x.NF
    o : ONote
    inst✝ : o.NF
    ⊢ Eq (x.scale o).repr (HMul.hMul (HPow.hPow Ordinal.omega0 x.repr) o.repr)
  -/
  simp only [scale_eq_mul, repr_mul, repr, PNat.one_coe, Nat.cast_one, mul_one, add_zero]
  /-
    🎉 no goals
  -/


theorem nf_repr_split {o o' m} [NF o] (h : split o = (o', m)) : NF o' ∧ repr o = repr o' + m := by
  /-
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := o', snd := m }
    ⊢ And o'.NF (Eq o.repr (HAdd.hAdd o'.repr ↑m))
  -/
  cases' e : split' o with a n
  /-
    case mk
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := o', snd := m }
    a : ONote
    n : Nat
    e : Eq o.split' { fst := a, snd := n }
    ⊢ And o'.NF (Eq o.repr (HAdd.hAdd o'.repr ↑m))
  -/
  cases' nf_repr_split' e with s₁ s₂
  /-
    case mk.intro
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := o', snd := m }
    a : ONote
    n : Nat
    e : Eq o.split' { fst := a, snd := n }
    s₁ : a.NF
    s₂ : Eq o.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a.repr) ↑n)
    ⊢ And o'.NF (Eq o.repr (HAdd.hAdd o'.repr ↑m))
  -/
  rw [split_eq_scale_split' e] at h
  /-
    case mk.intro
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    a : ONote
    n : Nat
    h : Eq { fst := ONote.scale 1 a, snd := n } { fst := o', snd := m }
    e : Eq o.split' { fst := a, snd := n }
    s₁ : a.NF
    s₂ : Eq o.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a.repr) ↑n)
    ⊢ And o'.NF (Eq o.repr (HAdd.hAdd o'.repr ↑m))
  -/
  injection h; substs o' n
  simp only [repr_scale, repr, opow_zero, Nat.succPNat_coe, Nat.cast_one, mul_one, add_zero,
    opow_one, s₂.symm, and_true]
  /-
    case mk.intro
    o : ONote
    m : Nat
    inst✝ : o.NF
    a : ONote
    s₁ : a.NF
    e : Eq o.split' { fst := a, snd := m }
    s₂ : Eq o.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 a.repr) ↑m)
    ⊢ (ONote.scale 1 a).NF
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem split_dvd {o o' m} [NF o] (h : split o = (o', m)) : ω ∣ repr o' := by
  /-
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := o', snd := m }
    ⊢ Dvd.dvd Ordinal.omega0 o'.repr
  -/
  cases' e : split' o with a n
  /-
    case mk
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := o', snd := m }
    a : ONote
    n : Nat
    e : Eq o.split' { fst := a, snd := n }
    ⊢ Dvd.dvd Ordinal.omega0 o'.repr
  -/
  rw [split_eq_scale_split' e] at h
  /-
    case mk
    o o' : ONote
    m : Nat
    inst✝ : o.NF
    a : ONote
    n : Nat
    h : Eq { fst := ONote.scale 1 a, snd := n } { fst := o', snd := m }
    e : Eq o.split' { fst := a, snd := n }
    ⊢ Dvd.dvd Ordinal.omega0 o'.repr
  -/
  injection h; subst o'
  /-
    case mk
    o : ONote
    m : Nat
    inst✝ : o.NF
    a : ONote
    n : Nat
    e : Eq o.split' { fst := a, snd := n }
    snd_eq✝ : Eq n m
    ⊢ Dvd.dvd Ordinal.omega0 (ONote.scale 1 a).repr
  -/
  cases nf_repr_split' e; simp
                          /-
                            🎉 no goals
                          -/


theorem split_add_lt {o e n a m} [NF o] (h : split o = (oadd e n a, m)) :
    repr a + m < ω ^ repr e := by
  /-
    o e : ONote
    n : PNat
    a : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := e.oadd n a, snd := m }
    ⊢ LT.lt (HAdd.hAdd a.repr ↑m) (HPow.hPow Ordinal.omega0 e.repr)
  -/
  cases' nf_repr_split h with h₁ h₂
  /-
    case intro
    o e : ONote
    n : PNat
    a : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := e.oadd n a, snd := m }
    h₁ : (e.oadd n a).NF
    h₂ : Eq o.repr (HAdd.hAdd (e.oadd n a).repr ↑m)
    ⊢ LT.lt (HAdd.hAdd a.repr ↑m) (HPow.hPow Ordinal.omega0 e.repr)
  -/
  cases' h₁.of_dvd_omega0 (split_dvd h) with e0 d
  /-
    case intro.intro
    o e : ONote
    n : PNat
    a : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := e.oadd n a, snd := m }
    h₁ : (e.oadd n a).NF
    h₂ : Eq o.repr (HAdd.hAdd (e.oadd n a).repr ↑m)
    e0 : Ne e.repr 0
    d : Dvd.dvd Ordinal.omega0 a.repr
    ⊢ LT.lt (HAdd.hAdd a.repr ↑m) (HPow.hPow Ordinal.omega0 e.repr)
  -/
  apply principal_add_omega0_opow _ h₁.snd'.repr_lt (lt_of_lt_of_le (nat_lt_omega0 _) _)
  /-
    o e : ONote
    n : PNat
    a : ONote
    m : Nat
    inst✝ : o.NF
    h : Eq o.split { fst := e.oadd n a, snd := m }
    h₁ : (e.oadd n a).NF
    h₂ : Eq o.repr (HAdd.hAdd (e.oadd n a).repr ↑m)
    e0 : Ne e.repr 0
    d : Dvd.dvd Ordinal.omega0 a.repr
    ⊢ LE.le Ordinal.omega0 (HPow.hPow Ordinal.omega0 e.repr)
  -/
  simpa using opow_le_opow_right omega0_pos (one_le_iff_ne_zero.2 e0)
  /-
    🎉 no goals
  -/


@[simp]
                                                             /-
                                                               n : Nat
                                                               o : ONote
                                                               ⊢ Eq (o.mulNat n) (HMul.hMul o ↑n)
                                                             -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
theorem mulNat_eq_mul (n o) : mulNat o n = o * ofNat n := by cases o <;> cases n <;> rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


                                                          /-
                                                            o : ONote
                                                            inst✝ : o.NF
                                                            n : Nat
                                                            ⊢ (o.mulNat n).NF
                                                          -/
instance nf_mulNat (o) [NF o] (n) : NF (mulNat o n) := by simpa using ONote.mul_nf o (ofNat n)
                                                          /-
                                                            🎉 no goals
                                                          -/


instance nf_opowAux (e a0 a) [NF e] [NF a0] [NF a] : ∀ k m, NF (opowAux e a0 a k m) := by
  /-
    e a0 a : ONote
    inst✝² : e.NF
    inst✝¹ : a0.NF
    inst✝ : a.NF
    ⊢ ∀ (k m : Nat), (e.opowAux a0 a k m).NF
  -/
  intro k m
  /-
    e a0 a : ONote
    inst✝² : e.NF
    inst✝¹ : a0.NF
    inst✝ : a.NF
    k m : Nat
    ⊢ (e.opowAux a0 a k m).NF
  -/
  unfold opowAux
  /-
    e a0 a : ONote
    inst✝² : e.NF
    inst✝¹ : a0.NF
    inst✝ : a.NF
    k m : Nat
    ⊢ (ONote.opowAux.match_1 (fun x x => ONote) k m (fun x => 0) (fun m => e.oadd  …
  -/
  cases' m with m m
    /-
      case zero
      e a0 a : ONote
      inst✝² : e.NF
      inst✝¹ : a0.NF
      inst✝ : a.NF
      k : Nat
      ⊢ (ONote.opowAux.match_1 (fun x x => ONote) k 0 (fun x => 0) (fun m => e.oadd  …
    -/
                /-
                  🎉 no goals
                -/
  · cases k <;> exact NF.zero
                /-
                  🎉 no goals
                -/
  /-
    case succ
    e a0 a : ONote
    inst✝² : e.NF
    inst✝¹ : a0.NF
    inst✝ : a.NF
    k m : Nat
    ⊢ (ONote.opowAux.match_1 (fun x x => ONote) k (HAdd.hAdd m 1) (fun x => 0) (fu …
  -/
  cases' k with k k
    /-
      case succ.zero
      e a0 a : ONote
      inst✝² : e.NF
      inst✝¹ : a0.NF
      inst✝ : a.NF
      m : Nat
      ⊢ (ONote.opowAux.match_1 (fun x x => ONote) 0 (HAdd.hAdd m 1) (fun x => 0) (fu …
    -/
  · exact NF.oadd_zero _ _
    /-
      🎉 no goals
    -/
    /-
      case succ.succ
      e a0 a : ONote
      inst✝² : e.NF
      inst✝¹ : a0.NF
      inst✝ : a.NF
      m k : Nat
      ⊢ (ONote.opowAux.match_1 (fun x x => ONote) (HAdd.hAdd k 1) (HAdd.hAdd m 1) (f …
    -/
  · haveI := nf_opowAux e a0 a k
    /-
      case succ.succ
      e a0 a : ONote
      inst✝² : e.NF
      inst✝¹ : a0.NF
      inst✝ : a.NF
      m k : Nat
      this : ∀ (m : Nat), (e.opowAux a0 a k m).NF
      ⊢ (ONote.opowAux.match_1 (fun x x => ONote) (HAdd.hAdd k 1) (HAdd.hAdd m 1) (f …
    -/
    simp only [Nat.succ_ne_zero m, IsEmpty.forall_iff, mulNat_eq_mul]; infer_instance
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


instance nf_opow (o₁ o₂) [NF o₁] [NF o₂] : NF (o₁ ^ o₂) := by
  /-
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    ⊢ (HPow.hPow o₁ o₂).NF
  -/
  cases' e₁ : split o₁ with a m
  /-
    case mk
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    a : ONote
    m : Nat
    e₁ : Eq o₁.split { fst := a, snd := m }
    ⊢ (HPow.hPow o₁ o₂).NF
  -/
  have na := (nf_repr_split e₁).1
  /-
    case mk
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    a : ONote
    m : Nat
    e₁ : Eq o₁.split { fst := a, snd := m }
    na : a.NF
    ⊢ (HPow.hPow o₁ o₂).NF
  -/
  cases' e₂ : split' o₂ with b' k
  /-
    case mk.mk
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    a : ONote
    m : Nat
    e₁ : Eq o₁.split { fst := a, snd := m }
    na : a.NF
    b' : ONote
    k : Nat
    e₂ : Eq o₂.split' { fst := b', snd := k }
    ⊢ (HPow.hPow o₁ o₂).NF
  -/
  haveI := (nf_repr_split' e₂).1
  /-
    case mk.mk
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    a : ONote
    m : Nat
    e₁ : Eq o₁.split { fst := a, snd := m }
    na : a.NF
    b' : ONote
    k : Nat
    e₂ : Eq o₂.split' { fst := b', snd := k }
    this : b'.NF
    ⊢ (HPow.hPow o₁ o₂).NF
  -/
  cases' a with a0 n a'
    /-
      case mk.mk.zero
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      this : b'.NF
      e₁ : Eq o₁.split { fst := ONote.zero, snd := m }
      na : ONote.zero.NF
      ⊢ (HPow.hPow o₁ o₂).NF
    -/
  · cases' m with m
      /-
        case mk.mk.zero.zero
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        b' : ONote
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := k }
        this : b'.NF
        na : ONote.zero.NF
        e₁ : Eq o₁.split { fst := ONote.zero, snd := 0 }
        ⊢ (HPow.hPow o₁ o₂).NF
      -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    · by_cases o₂ = 0 <;> simp only [(· ^ ·), Pow.pow, pow, opow, opowAux2, *] <;> decide
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
      /-
        case mk.mk.zero.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        b' : ONote
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := k }
        this : b'.NF
        na : ONote.zero.NF
        m : Nat
        e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
        ⊢ (HPow.hPow o₁ o₂).NF
      -/
    · by_cases m = 0
        /-
          case pos
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          this : b'.NF
          na : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          h✝ : Eq m 0
          ⊢ (HPow.hPow o₁ o₂).NF
        -/
      · simp only [(· ^ ·), Pow.pow, pow, opow, opowAux2, *, zero_def]
        /-
          case pos
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          this : b'.NF
          na : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          h✝ : Eq m 0
          ⊢ ONote.NF 1
        -/
        decide
        /-
          🎉 no goals
        -/
        /-
          case neg
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          this : b'.NF
          na : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          h✝ : Not (Eq m 0)
          ⊢ (HPow.hPow o₁ o₂).NF
        -/
      · simp only [(· ^ ·), Pow.pow, pow, opow, opowAux2, mulNat_eq_mul, ofNat, *]
        /-
          case neg
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          this : b'.NF
          na : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          h✝ : Not (Eq m 0)
          ⊢ (b'.oadd (Monoid.npow k m.succPNat) 0).NF
        -/
        infer_instance
        /-
          🎉 no goals
        -/
    /-
      case mk.mk.oadd
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      this : b'.NF
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      na : (a0.oadd n a').NF
      ⊢ (HPow.hPow o₁ o₂).NF
    -/
  · simp only [(· ^ ·), Pow.pow, opow, opowAux2, e₁, split_eq_scale_split' e₂, mulNat_eq_mul]
    /-
      case mk.mk.oadd
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      this : b'.NF
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      na : (a0.oadd n a').NF
      ⊢ (ONote.opowAux2.match_1 (fun x => ONote) { fst := ONote.scale 1 b', snd := k …
    -/
    have := na.fst
    /-
      case mk.mk.oadd
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      this✝ : b'.NF
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      na : (a0.oadd n a').NF
      this : a0.NF
      ⊢ (ONote.opowAux2.match_1 (fun x => ONote) { fst := ONote.scale 1 b', snd := k …
    -/
    cases' k with k
      /-
        case mk.mk.oadd.zero
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        b' : ONote
        this✝ : b'.NF
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        na : (a0.oadd n a').NF
        this : a0.NF
        e₂ : Eq o₂.split' { fst := b', snd := 0 }
        ⊢ (ONote.opowAux2.match_1 (fun x => ONote) { fst := ONote.scale 1 b', snd := 0 …
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.oadd.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        b' : ONote
        this✝ : b'.NF
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        na : (a0.oadd n a').NF
        this : a0.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        ⊢ (ONote.opowAux2.match_1 (fun x => ONote) { fst := ONote.scale 1 b', snd := H …
      -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
    · cases k <;> cases m <;> infer_instance
                              /-
                                🎉 no goals
                              -/


theorem scale_opowAux (e a0 a : ONote) [NF e] [NF a0] [NF a] :
    ∀ k m, repr (opowAux e a0 a k m) = ω ^ repr e * repr (opowAux 0 a0 a k m)
               /-
                 e a0 a : ONote
                 inst✝² : e.NF
                 inst✝¹ : a0.NF
                 inst✝ : a.NF
                 m : Nat
                 ⊢ Eq (e.opowAux a0 a 0 m).repr (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) (O …
               -/
                           /-
                             🎉 no goals
                           -/
  | 0, m => by cases m <;> simp [opowAux]
                           /-
                             🎉 no goals
                           -/
  | k + 1, m => by
    /-
      e a0 a : ONote
      inst✝² : e.NF
      inst✝¹ : a0.NF
      inst✝ : a.NF
      k m : Nat
      ⊢ Eq (e.opowAux a0 a (HAdd.hAdd k 1) m).repr (HMul.hMul (HPow.hPow Ordinal.ome …
    -/
    by_cases h : m = 0
      /-
        case pos
        e a0 a : ONote
        inst✝² : e.NF
        inst✝¹ : a0.NF
        inst✝ : a.NF
        k m : Nat
        h : Eq m 0
        ⊢ Eq (e.opowAux a0 a (HAdd.hAdd k 1) m).repr (HMul.hMul (HPow.hPow Ordinal.ome …
      -/
    · simp [h, opowAux, mul_add, opow_add, mul_assoc, scale_opowAux _ _ _ k]
      /-
        🎉 no goals
      -/
    · -- Porting note: rewrote proof
      /-
        case neg
        e a0 a : ONote
        inst✝² : e.NF
        inst✝¹ : a0.NF
        inst✝ : a.NF
        k m : Nat
        h : Not (Eq m 0)
        ⊢ Eq (e.opowAux a0 a (HAdd.hAdd k 1) m).repr (HMul.hMul (HPow.hPow Ordinal.ome …
      -/
      rw [opowAux]; swap
        /-
          case neg.x_2
          e a0 a : ONote
          inst✝² : e.NF
          inst✝¹ : a0.NF
          inst✝ : a.NF
          k m : Nat
          h : Not (Eq m 0)
          ⊢ Eq m 0 → False
        -/
      · assumption
        /-
          🎉 no goals
        -/
      /-
        case neg
        e a0 a : ONote
        inst✝² : e.NF
        inst✝¹ : a0.NF
        inst✝ : a.NF
        k m : Nat
        h : Not (Eq m 0)
        ⊢ Eq (HAdd.hAdd ((HAdd.hAdd e (a0.mulNat k)).scale a) (e.opowAux a0 a k m)).re …
      -/
      rw [opowAux]; swap
        /-
          case neg.x_2
          e a0 a : ONote
          inst✝² : e.NF
          inst✝¹ : a0.NF
          inst✝ : a.NF
          k m : Nat
          h : Not (Eq m 0)
          ⊢ Eq m 0 → False
        -/
      · assumption
        /-
          🎉 no goals
        -/
      /-
        case neg
        e a0 a : ONote
        inst✝² : e.NF
        inst✝¹ : a0.NF
        inst✝ : a.NF
        k m : Nat
        h : Not (Eq m 0)
        ⊢ Eq (HAdd.hAdd ((HAdd.hAdd e (a0.mulNat k)).scale a) (e.opowAux a0 a k m)).re …
      -/
      rw [repr_add, repr_scale, scale_opowAux _ _ _ k]
      /-
        case neg
        e a0 a : ONote
        inst✝² : e.NF
        inst✝¹ : a0.NF
        inst✝ : a.NF
        k m : Nat
        h : Not (Eq m 0)
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 (HAdd.hAdd e (a0.mulNat k …
      -/
      simp only [repr_add, repr_scale, opow_add, mul_assoc, zero_add, mul_add]
      /-
        🎉 no goals
      -/


theorem repr_opow_aux₁ {e a} [Ne : NF e] [Na : NF a] {a' : Ordinal} (e0 : repr e ≠ 0)
    (h : a' < (ω : Ordinal.{0}) ^ repr e) (aa : repr a = a') (n : ℕ+) :
    ((ω : Ordinal.{0}) ^ repr e * (n : ℕ) + a') ^ (ω : Ordinal.{0}) =
      (ω ^ repr e) ^ (ω : Ordinal.{0}) := by
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    a' : Ordinal.{0}
    e0 : _root_.Ne e.repr 0
    h : LT.lt a' (HPow.hPow Ordinal.omega0 e.repr)
    aa : Eq a.repr a'
    n : PNat
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a …
  -/
  subst aa
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a …
  -/
  have No := Ne.oadd n (Na.below_of_lt' h)
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a …
  -/
  have := omega0_le_oadd e n a
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this : LE.le (HPow.hPow Ordinal.omega0 e.repr) (e.oadd n a).repr
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a …
  -/
  rw [repr] at this
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hPo …
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a …
  -/
  refine le_antisymm ?_ (opow_le_opow_left _ this)
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hPo …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n …
  -/
  apply (opow_le_of_limit ((opow_pos _ omega0_pos).trans_le this).ne' isLimit_omega0).2
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hPo …
    ⊢ ∀ (b' : Ordinal.{0}), LT.lt b' Ordinal.omega0 → LE.le (HPow.hPow (HAdd.hAdd  …
  -/
  intro b l
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hPo …
    b : Ordinal.{0}
    l : LT.lt b Ordinal.omega0
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n …
  -/
  have := (No.below_of_lt (lt_succ _)).repr_lt
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
    b : Ordinal.{0}
    l : LT.lt b Ordinal.omega0
    this : LT.lt (e.oadd n a).repr (HPow.hPow Ordinal.omega0 (Order.succ e.repr))
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n …
  -/
  rw [repr] at this
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
    b : Ordinal.{0}
    l : LT.lt b Ordinal.omega0
    this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
    ⊢ LE.le (HPow.hPow (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n …
  -/
  apply (opow_le_opow_left b <| this.le).trans
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
    b : Ordinal.{0}
    l : LT.lt b Ordinal.omega0
    this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
    ⊢ LE.le (HPow.hPow (HPow.hPow Ordinal.omega0 (Order.succ e.repr)) b) (HPow.hPo …
  -/
  rw [← opow_mul, ← opow_mul]
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
    b : Ordinal.{0}
    l : LT.lt b Ordinal.omega0
    this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
    ⊢ LE.le (HPow.hPow Ordinal.omega0 (HMul.hMul (Order.succ e.repr) b)) (HPow.hPo …
  -/
  apply opow_le_opow_right omega0_pos
  /-
    e a : ONote
    Ne : e.NF
    Na : a.NF
    e0 : _root_.Ne e.repr 0
    n : PNat
    h : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
    No : (e.oadd n a).NF
    this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
    b : Ordinal.{0}
    l : LT.lt b Ordinal.omega0
    this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
    ⊢ LE.le (HMul.hMul (Order.succ e.repr) b) (HMul.hMul e.repr Ordinal.omega0)
  -/
  rcases le_or_lt ω (repr e) with h | h
    /-
      case inl
      e a : ONote
      Ne : e.NF
      Na : a.NF
      e0 : _root_.Ne e.repr 0
      n : PNat
      h✝ : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
      No : (e.oadd n a).NF
      this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
      b : Ordinal.{0}
      l : LT.lt b Ordinal.omega0
      this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
      h : LE.le Ordinal.omega0 e.repr
      ⊢ LE.le (HMul.hMul (Order.succ e.repr) b) (HMul.hMul e.repr Ordinal.omega0)
    -/
  · apply (mul_le_mul_left' (le_succ b) _).trans
    rw [← add_one_eq_succ, add_mul_succ _ (one_add_of_omega0_le h), add_one_eq_succ, succ_le_iff,
      Ordinal.mul_lt_mul_iff_left (Ordinal.pos_iff_ne_zero.2 e0)]
    /-
      case inl
      e a : ONote
      Ne : e.NF
      Na : a.NF
      e0 : _root_.Ne e.repr 0
      n : PNat
      h✝ : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
      No : (e.oadd n a).NF
      this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
      b : Ordinal.{0}
      l : LT.lt b Ordinal.omega0
      this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
      h : LE.le Ordinal.omega0 e.repr
      ⊢ LT.lt (Order.succ b) Ordinal.omega0
    -/
    exact isLimit_omega0.succ_lt l
    /-
      🎉 no goals
    -/
    /-
      case inr
      e a : ONote
      Ne : e.NF
      Na : a.NF
      e0 : _root_.Ne e.repr 0
      n : PNat
      h✝ : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
      No : (e.oadd n a).NF
      this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
      b : Ordinal.{0}
      l : LT.lt b Ordinal.omega0
      this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
      h : LT.lt e.repr Ordinal.omega0
      ⊢ LE.le (HMul.hMul (Order.succ e.repr) b) (HMul.hMul e.repr Ordinal.omega0)
    -/
  · apply (principal_mul_omega0 (isLimit_omega0.succ_lt h) l).le.trans
    /-
      case inr
      e a : ONote
      Ne : e.NF
      Na : a.NF
      e0 : _root_.Ne e.repr 0
      n : PNat
      h✝ : LT.lt a.repr (HPow.hPow Ordinal.omega0 e.repr)
      No : (e.oadd n a).NF
      this✝ : LE.le (HPow.hPow Ordinal.omega0 e.repr) (HAdd.hAdd (HMul.hMul (HPow.hP …
      b : Ordinal.{0}
      l : LT.lt b Ordinal.omega0
      this : LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 e.repr) ↑↑n) a.re …
      h : LT.lt e.repr Ordinal.omega0
      ⊢ LE.le Ordinal.omega0 (HMul.hMul e.repr Ordinal.omega0)
    -/
    simpa using mul_le_mul_right' (one_le_iff_ne_zero.2 e0) ω
    /-
      🎉 no goals
    -/


set_option linter.unusedVariables false in
theorem repr_opow_aux₂ {a0 a'} [N0 : NF a0] [Na' : NF a'] (m : ℕ) (d : ω ∣ repr a')
    (e0 : repr a0 ≠ 0) (h : repr a' + m < (ω ^ repr a0)) (n : ℕ+) (k : ℕ) :
    let R := repr (opowAux 0 a0 (oadd a0 n a' * ofNat m) k m)
    (k ≠ 0 → R < ((ω ^ repr a0) ^ succ (k : Ordinal))) ∧
      ((ω ^ repr a0) ^ (k : Ordinal)) * ((ω ^ repr a0) * (n : ℕ) + repr a') + R =
        ((ω ^ repr a0) * (n : ℕ) + repr a' + m) ^ succ (k : Ordinal) := by
  /-
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    k : Nat
    ⊢ let R := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr;
      And (Ne k 0 → LT.lt R (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order.s …
  -/
  intro R'
  haveI No : NF (oadd a0 n a') :=
    N0.oadd n (Na'.below_of_lt' <| lt_of_le_of_lt (le_add_right _ _) h)
  /-
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    k : Nat
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    No : (a0.oadd n a').NF
    ⊢ And (Ne k 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order. …
  -/
  induction' k with k IH
    /-
      case zero
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0 : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) 0 m).repr
      ⊢ And (Ne 0 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order. …
    -/
                /-
                  🎉 no goals
                -/
  · cases m <;> simp [R', opowAux]
                /-
                  🎉 no goals
                -/
  -- rename R => R'
  /-
    case succ
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    IH :
      let R' := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr;
      And (Ne k 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order. …
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    ⊢ And (Ne (HAdd.hAdd k 1) 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0 …
  -/
  let R := repr (opowAux 0 a0 (oadd a0 n a' * ofNat m) k m)
  /-
    case succ
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    IH :
      let R' := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr;
      And (Ne k 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order. …
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    ⊢ And (Ne (HAdd.hAdd k 1) 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0 …
  -/
  let ω0 := ω ^ repr a0
  /-
    case succ
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    IH :
      let R' := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr;
      And (Ne k 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order. …
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
    ⊢ And (Ne (HAdd.hAdd k 1) 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0 …
  -/
  let α' := ω0 * n + repr a'
  change (k ≠ 0 → R < (ω0 ^ succ (k : Ordinal))) ∧ (ω0 ^ (k : Ordinal)) * α' + R
    = (α' + m) ^ (succ ↑k : Ordinal) at IH
  have RR : R' = ω0 ^ (k : Ordinal) * (α' * m) + R := by
    by_cases h : m = 0
    · simp only [R, R', h, ONote.ofNat, Nat.cast_zero, zero_add, ONote.repr, mul_zero,
        ONote.opowAux, add_zero]
    · simp only [α', ω0, R, R', ONote.repr_scale, ONote.repr, ONote.mulNat_eq_mul, ONote.opowAux,
        ONote.repr_ofNat, ONote.repr_mul, ONote.repr_add, Ordinal.opow_mul, ONote.zero_add]
  /-
    case succ
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
    α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
    IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
    RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
    ⊢ And (Ne (HAdd.hAdd k 1) 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0 …
  -/
  have α0 : 0 < α' := by simpa [lt_def, repr] using oadd_pos a0 n a'
  /-
    case succ
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
    α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
    IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
    RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
    α0 : LT.lt 0 α'
    ⊢ And (Ne (HAdd.hAdd k 1) 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0 …
  -/
  have ω00 : 0 < ω0 ^ (k : Ordinal) := opow_pos _ (opow_pos _ omega0_pos)
  have Rl : R < ω ^ (repr a0 * succ ↑k) := by
    by_cases k0 : k = 0
    · simp only [k0, Nat.cast_zero, succ_zero, mul_one, R]
      refine lt_of_lt_of_le ?_ (opow_le_opow_right omega0_pos (one_le_iff_ne_zero.2 e0))
      cases' m with m <;> simp [opowAux, omega0_pos]
      rw [← add_one_eq_succ, ← Nat.cast_succ]
      apply nat_lt_omega0
    · rw [opow_mul]
      exact IH.1 k0
  /-
    case succ
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
    α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
    IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
    RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
    α0 : LT.lt 0 α'
    ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
    Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
    ⊢ And (Ne (HAdd.hAdd k 1) 0 → LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0 …
  -/
  refine ⟨fun _ => ?_, ?_⟩
    /-
      case succ.refine_1
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0 : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      k : Nat
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
      R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
      ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
      α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
      IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
      RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
      α0 : LT.lt 0 α'
      ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
      Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
      x✝ : Ne (HAdd.hAdd k 1) 0
      ⊢ LT.lt R' (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order.succ ↑(HAdd.hA …
    -/
  · rw [RR, ← opow_mul _ _ (succ k.succ)]
    /-
      case succ.refine_1
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0 : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      k : Nat
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
      R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
      ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
      α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
      IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
      RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
      α0 : LT.lt 0 α'
      ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
      Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
      x✝ : Ne (HAdd.hAdd k 1) 0
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R) (HPow.hP …
    -/
    have e0 := Ordinal.pos_iff_ne_zero.2 e0
    /-
      case succ.refine_1
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0✝ : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      k : Nat
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
      R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
      ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
      α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
      IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
      RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
      α0 : LT.lt 0 α'
      ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
      Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
      x✝ : Ne (HAdd.hAdd k 1) 0
      e0 : LT.lt 0 a0.repr
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R) (HPow.hP …
    -/
    have rr0 : 0 < repr a0 + repr a0 := lt_of_lt_of_le e0 (le_add_left _ _)
    /-
      case succ.refine_1
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0✝ : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      k : Nat
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
      R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
      ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
      α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
      IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
      RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
      α0 : LT.lt 0 α'
      ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
      Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
      x✝ : Ne (HAdd.hAdd k 1) 0
      e0 : LT.lt 0 a0.repr
      rr0 : LT.lt 0 (HAdd.hAdd a0.repr a0.repr)
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R) (HPow.hP …
    -/
    apply principal_add_omega0_opow
    · simp only [Nat.succ_eq_add_one, Nat.cast_add, Nat.cast_one, add_one_eq_succ,
        opow_mul, opow_succ, mul_assoc]
      /-
        case succ.refine_1.a
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0✝ : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        x✝ : Ne (HAdd.hAdd k 1) 0
        e0 : LT.lt 0 a0.repr
        rr0 : LT.lt 0 (HAdd.hAdd a0.repr a0.repr)
        ⊢ LT.lt (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) (HMul.hMul (HPow.hPow  …
      -/
      rw [Ordinal.mul_lt_mul_iff_left ω00, ← Ordinal.opow_add]
      /-
        case succ.refine_1.a
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0✝ : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        x✝ : Ne (HAdd.hAdd k 1) 0
        e0 : LT.lt 0 a0.repr
        rr0 : LT.lt 0 (HAdd.hAdd a0.repr a0.repr)
        ⊢ LT.lt (HMul.hMul α' ↑m) (HPow.hPow Ordinal.omega0 (HAdd.hAdd a0.repr a0.repr))
      -/
      have : _ < ω ^ (repr a0 + repr a0) := (No.below_of_lt ?_).repr_lt
        /-
          case succ.refine_1.a.refine_2
          a0 a' : ONote
          N0 : a0.NF
          Na' : a'.NF
          m : Nat
          d : Dvd.dvd Ordinal.omega0 a'.repr
          e0✝ : Ne a0.repr 0
          h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
          n : PNat
          No : (a0.oadd n a').NF
          k : Nat
          R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
          R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
          ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
          α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
          IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
          RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
          α0 : LT.lt 0 α'
          ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
          Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
          x✝ : Ne (HAdd.hAdd k 1) 0
          e0 : LT.lt 0 a0.repr
          rr0 : LT.lt 0 (HAdd.hAdd a0.repr a0.repr)
          this : LT.lt (a0.oadd n a').repr (HPow.hPow Ordinal.omega0 (HAdd.hAdd a0.repr  …
          ⊢ LT.lt (HMul.hMul α' ↑m) (HPow.hPow Ordinal.omega0 (HAdd.hAdd a0.repr a0.repr))
        -/
      · exact mul_lt_omega0_opow rr0 this (nat_lt_omega0 _)
        /-
          🎉 no goals
        -/
        /-
          case succ.refine_1.a.refine_1
          a0 a' : ONote
          N0 : a0.NF
          Na' : a'.NF
          m : Nat
          d : Dvd.dvd Ordinal.omega0 a'.repr
          e0✝ : Ne a0.repr 0
          h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
          n : PNat
          No : (a0.oadd n a').NF
          k : Nat
          R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
          R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
          ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
          α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
          IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
          RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
          α0 : LT.lt 0 α'
          ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
          Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
          x✝ : Ne (HAdd.hAdd k 1) 0
          e0 : LT.lt 0 a0.repr
          rr0 : LT.lt 0 (HAdd.hAdd a0.repr a0.repr)
          ⊢ LT.lt a0.repr (HAdd.hAdd a0.repr a0.repr)
        -/
      · simpa using (add_lt_add_iff_left (repr a0)).2 e0
        /-
          🎉 no goals
        -/
    · exact
        lt_of_lt_of_le Rl
          (opow_le_opow_right omega0_pos <|
            mul_le_mul_left' (succ_le_succ_iff.2 (Nat.cast_le.2 (le_of_lt k.lt_succ_self))) _)
  calc
    (ω0 ^ (k.succ : Ordinal)) * α' + R'
    _ = (ω0 ^ succ (k : Ordinal)) * α' + ((ω0 ^ (k : Ordinal)) * α' * m + R) := by
        rw [natCast_succ, RR, ← mul_assoc]
    _ = ((ω0 ^ (k : Ordinal)) * α' + R) * α' + ((ω0 ^ (k : Ordinal)) * α' + R) * m := ?_
    _ = (α' + m) ^ succ (k.succ : Ordinal) := by rw [← mul_add, natCast_succ, opow_succ, IH.2]
  /-
    case succ.refine_2
    a0 a' : ONote
    N0 : a0.NF
    Na' : a'.NF
    m : Nat
    d : Dvd.dvd Ordinal.omega0 a'.repr
    e0 : Ne a0.repr 0
    h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
    n : PNat
    No : (a0.oadd n a').NF
    k : Nat
    R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
    R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
    ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
    α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
    IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
    RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
    α0 : LT.lt 0 α'
    ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
    Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 (Order.succ ↑k)) α') (HAdd.hAdd (HMul …
  -/
  congr 1
  · have αd : ω ∣ α' :=
      dvd_add (dvd_mul_of_dvd_left (by simpa using opow_dvd_opow ω (one_le_iff_ne_zero.2 e0)) _) d
    rw [mul_add (ω0 ^ (k : Ordinal)), add_assoc, ← mul_assoc, ← opow_succ,
      add_mul_limit _ (isLimit_iff_omega0_dvd.2 ⟨ne_of_gt α0, αd⟩), mul_assoc,
      @mul_omega0_dvd n (Nat.cast_pos'.2 n.pos) (nat_lt_omega0 _) _ αd]
    /-
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0 : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      k : Nat
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
      R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
      ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
      α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
      IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
      RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
      α0 : LT.lt 0 α'
      ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
      Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
      αd : Dvd.dvd Ordinal.omega0 α'
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) a'.repr) R) (HMul.hMul …
    -/
    apply @add_absorp _ (repr a0 * succ ↑k)
      /-
        case h₁
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        αd : Dvd.dvd Ordinal.omega0 α'
        ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) a'.repr) R) (HPow.hPow Ordinal …
      -/
    · refine principal_add_omega0_opow _ ?_ Rl
      /-
        case h₁
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        αd : Dvd.dvd Ordinal.omega0 α'
        ⊢ LT.lt (HMul.hMul (HPow.hPow ω0 ↑k) a'.repr) (HPow.hPow Ordinal.omega0 (HMul. …
      -/
      rw [opow_mul, opow_succ, Ordinal.mul_lt_mul_iff_left ω00]
      /-
        case h₁
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        αd : Dvd.dvd Ordinal.omega0 α'
        ⊢ LT.lt a'.repr (HPow.hPow Ordinal.omega0 a0.repr)
      -/
      exact No.snd'.repr_lt
      /-
        🎉 no goals
      -/
      /-
        case h₂
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        αd : Dvd.dvd Ordinal.omega0 α'
        ⊢ LE.le (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k))) (HMul.h …
      -/
    · have := mul_le_mul_left' (one_le_iff_pos.2 <| Nat.cast_pos'.2 n.pos) (ω0 ^ succ (k : Ordinal))
      /-
        case h₂
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        αd : Dvd.dvd Ordinal.omega0 α'
        this : LE.le (HMul.hMul (HPow.hPow ω0 (Order.succ ↑k)) 1) (HMul.hMul (HPow.hPo …
        ⊢ LE.le (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k))) (HMul.h …
      -/
      rw [opow_mul]
      /-
        case h₂
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        m : Nat
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        αd : Dvd.dvd Ordinal.omega0 α'
        this : LE.le (HMul.hMul (HPow.hPow ω0 (Order.succ ↑k)) 1) (HMul.hMul (HPow.hPo …
        ⊢ LE.le (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) (Order.succ ↑k)) (HMul.h …
      -/
      simpa [-opow_succ]
      /-
        🎉 no goals
      -/
    /-
      case succ.refine_2.e_a
      a0 a' : ONote
      N0 : a0.NF
      Na' : a'.NF
      m : Nat
      d : Dvd.dvd Ordinal.omega0 a'.repr
      e0 : Ne a0.repr 0
      h : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      n : PNat
      No : (a0.oadd n a').NF
      k : Nat
      R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) (HAdd.hA …
      R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑m) k m).repr
      ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
      α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
      IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
      RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑m)) R)
      α0 : LT.lt 0 α'
      ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
      Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow ω0 ↑k) α') ↑m) R) (HMul.hMul  …
    -/
  · cases m
      /-
        case succ.refine_2.e_a.zero
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        h : LT.lt (HAdd.hAdd a'.repr ↑0) (HPow.hPow Ordinal.omega0 a0.repr)
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑0) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑0) k 0).repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑0)) R)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow ω0 ↑k) α') ↑0) R) (HMul.hMul  …
      -/
    · have : R = 0 := by cases k <;> simp [R, opowAux]
      /-
        case succ.refine_2.e_a.zero
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        h : LT.lt (HAdd.hAdd a'.repr ↑0) (HPow.hPow Ordinal.omega0 a0.repr)
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑0) (HAdd.hA …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑0) k 0).repr
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑0)) R)
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        this : Eq R 0
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow ω0 ↑k) α') ↑0) R) (HMul.hMul  …
      -/
      simp [this]
      /-
        🎉 no goals
      -/
      /-
        case succ.refine_2.e_a.succ
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        n✝ : Nat
        h : LT.lt (HAdd.hAdd a'.repr ↑(HAdd.hAdd n✝ 1)) (HPow.hPow Ordinal.omega0 a0.r …
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd  …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd n …
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑(HAdd.hAdd n …
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow ω0 ↑k) α') ↑(HAdd.hAdd n✝ 1)) …
      -/
    · rw [natCast_succ, add_mul_succ]
      /-
        case succ.refine_2.e_a.succ.ba
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        n✝ : Nat
        h : LT.lt (HAdd.hAdd a'.repr ↑(HAdd.hAdd n✝ 1)) (HPow.hPow Ordinal.omega0 a0.r …
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd  …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd n …
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑(HAdd.hAdd n …
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        ⊢ Eq (HAdd.hAdd R (HMul.hMul (HPow.hPow ω0 ↑k) α')) (HMul.hMul (HPow.hPow ω0 ↑ …
      -/
      apply add_absorp Rl
      /-
        case succ.refine_2.e_a.succ.ba
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        n✝ : Nat
        h : LT.lt (HAdd.hAdd a'.repr ↑(HAdd.hAdd n✝ 1)) (HPow.hPow Ordinal.omega0 a0.r …
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd  …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd n …
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑(HAdd.hAdd n …
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        ⊢ LE.le (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k))) (HMul.h …
      -/
      rw [opow_mul, opow_succ]
      /-
        case succ.refine_2.e_a.succ.ba
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        n✝ : Nat
        h : LT.lt (HAdd.hAdd a'.repr ↑(HAdd.hAdd n✝ 1)) (HPow.hPow Ordinal.omega0 a0.r …
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd  …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd n …
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑(HAdd.hAdd n …
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        ⊢ LE.le (HMul.hMul (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) ↑k) (HPow.hPo …
      -/
      apply mul_le_mul_left'
      /-
        case succ.refine_2.e_a.succ.ba.bc
        a0 a' : ONote
        N0 : a0.NF
        Na' : a'.NF
        d : Dvd.dvd Ordinal.omega0 a'.repr
        e0 : Ne a0.repr 0
        n : PNat
        No : (a0.oadd n a').NF
        k : Nat
        ω0 : Ordinal.{0} := HPow.hPow Ordinal.omega0 a0.repr
        α' : Ordinal.{0} := HAdd.hAdd (HMul.hMul ω0 ↑↑n) a'.repr
        α0 : LT.lt 0 α'
        ω00 : LT.lt 0 (HPow.hPow ω0 ↑k)
        n✝ : Nat
        h : LT.lt (HAdd.hAdd a'.repr ↑(HAdd.hAdd n✝ 1)) (HPow.hPow Ordinal.omega0 a0.r …
        R' : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd  …
        R : Ordinal.{0} := (ONote.opowAux 0 a0 (HMul.hMul (a0.oadd n a') ↑(HAdd.hAdd n …
        IH : And (Ne k 0 → LT.lt R (HPow.hPow ω0 (Order.succ ↑k))) (Eq (HAdd.hAdd (HMu …
        RR : Eq R' (HAdd.hAdd (HMul.hMul (HPow.hPow ω0 ↑k) (HMul.hMul α' ↑(HAdd.hAdd n …
        Rl : LT.lt R (HPow.hPow Ordinal.omega0 (HMul.hMul a0.repr (Order.succ ↑k)))
        ⊢ LE.le (HPow.hPow Ordinal.omega0 a0.repr) α'
      -/
      simpa [repr] using omega0_le_oadd a0 n a'
      /-
        🎉 no goals
      -/


theorem repr_opow (o₁ o₂) [NF o₁] [NF o₂] : repr (o₁ ^ o₂) = repr o₁ ^ repr o₂ := by
  /-
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
  -/
  cases' e₁ : split o₁ with a m
  /-
    case mk
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    a : ONote
    m : Nat
    e₁ : Eq o₁.split { fst := a, snd := m }
    ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
  -/
  cases' nf_repr_split e₁ with N₁ r₁
  /-
    case mk.intro
    o₁ o₂ : ONote
    inst✝¹ : o₁.NF
    inst✝ : o₂.NF
    a : ONote
    m : Nat
    e₁ : Eq o₁.split { fst := a, snd := m }
    N₁ : a.NF
    r₁ : Eq o₁.repr (HAdd.hAdd a.repr ↑m)
    ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
  -/
  cases' a with a0 n a'
    /-
      case mk.intro.zero
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      e₁ : Eq o₁.split { fst := ONote.zero, snd := m }
      N₁ : ONote.zero.NF
      r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑m)
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
  · cases' m with m
      /-
        case mk.intro.zero.zero
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        e₁ : Eq o₁.split { fst := ONote.zero, snd := 0 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑0)
        ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
      -/
                              /-
                                🎉 no goals
                              -/
    · by_cases h : o₂ = 0 <;> simp [opow_def, opowAux2, opow, e₁, h, r₁]
      /-
        case neg
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        e₁ : Eq o₁.split { fst := ONote.zero, snd := 0 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑0)
        h : Not (Eq o₂ 0)
        ⊢ Eq 0 (HPow.hPow 0 o₂.repr)
      -/
      have := mt repr_inj.1 h
      /-
        case neg
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        e₁ : Eq o₁.split { fst := ONote.zero, snd := 0 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑0)
        h : Not (Eq o₂ 0)
        this : Not (Eq o₂.repr (ONote.repr 0))
        ⊢ Eq 0 (HPow.hPow 0 o₂.repr)
      -/
      rw [zero_opow this]
      /-
        🎉 no goals
      -/
      /-
        case mk.intro.zero.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        m : Nat
        e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
        ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
      -/
    · cases' e₂ : split' o₂ with b' k
      /-
        case mk.intro.zero.succ.mk
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        m : Nat
        e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
        b' : ONote
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := k }
        ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
      -/
      cases' nf_repr_split' e₂ with _ r₂
      /-
        case mk.intro.zero.succ.mk.intro
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        m : Nat
        e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
        b' : ONote
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := k }
        left✝ : b'.NF
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
        ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
      -/
      by_cases h : m = 0
        /-
          case pos
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          N₁ : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          left✝ : b'.NF
          r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
          h : Eq m 0
          ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
        -/
      · simp [opow_def, opow, e₁, h, r₁, e₂, r₂]
        /-
          🎉 no goals
        -/
      simp only [opow_def, opowAux2, opow, e₁, h, r₁, e₂, r₂, repr,
          opow_zero, Nat.succPNat_coe, Nat.cast_succ, Nat.cast_zero, _root_.zero_add, mul_one,
          add_zero, one_opow, npow_eq_pow]
      /-
        case neg
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        N₁ : ONote.zero.NF
        m : Nat
        e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
        r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
        b' : ONote
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := k }
        left✝ : b'.NF
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
        h : Not (Eq m 0)
        ⊢ Eq (HMul.hMul (HPow.hPow Ordinal.omega0 b'.repr) ↑↑(HPow.hPow m.succPNat k)) …
      -/
      rw [opow_add, opow_mul, opow_omega0, add_one_eq_succ]
        /-
          case neg
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          N₁ : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          left✝ : b'.NF
          r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
          h : Not (Eq m 0)
          ⊢ Eq (HMul.hMul (HPow.hPow Ordinal.omega0 b'.repr) ↑↑(HPow.hPow m.succPNat k)) …
        -/
      · congr
        conv_lhs =>
          dsimp [(· ^ ·)]
          simp [Pow.pow, opow, Ordinal.succ_ne_zero]
        /-
          case neg.e_a
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          N₁ : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          left✝ : b'.NF
          r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
          h : Not (Eq m 0)
          ⊢ Eq (HPow.hPow (Order.succ ↑m) k) (HPow.hPow (Order.succ ↑m) ↑k)
        -/
        rw [opow_natCast]
        /-
          🎉 no goals
        -/
        /-
          case neg.a1
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          N₁ : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          left✝ : b'.NF
          r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
          h : Not (Eq m 0)
          ⊢ LT.lt 1 (HAdd.hAdd (↑m) 1)
        -/
      · simpa [Nat.one_le_iff_ne_zero]
        /-
          🎉 no goals
        -/
        /-
          case neg.h
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          N₁ : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          left✝ : b'.NF
          r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
          h : Not (Eq m 0)
          ⊢ LT.lt (HAdd.hAdd (↑m) 1) Ordinal.omega0
        -/
      · rw [← Nat.cast_succ, lt_omega0]
        /-
          case neg.h
          o₁ o₂ : ONote
          inst✝¹ : o₁.NF
          inst✝ : o₂.NF
          N₁ : ONote.zero.NF
          m : Nat
          e₁ : Eq o₁.split { fst := ONote.zero, snd := HAdd.hAdd m 1 }
          r₁ : Eq o₁.repr (HAdd.hAdd ONote.zero.repr ↑(HAdd.hAdd m 1))
          b' : ONote
          k : Nat
          e₂ : Eq o₂.split' { fst := b', snd := k }
          left✝ : b'.NF
          r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
          h : Not (Eq m 0)
          ⊢ Exists fun n => Eq ↑m.succ ↑n
        -/
        exact ⟨_, rfl⟩
        /-
          🎉 no goals
        -/
    /-
      case mk.intro.oadd
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
  · haveI := N₁.fst
    /-
      case mk.intro.oadd
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this : a0.NF
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
    haveI := N₁.snd
    /-
      case mk.intro.oadd
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this✝ : a0.NF
      this : a'.NF
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
    cases' N₁.of_dvd_omega0 (split_dvd e₁) with a00 ad
    /-
      case mk.intro.oadd.intro
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this✝ : a0.NF
      this : a'.NF
      a00 : Ne a0.repr 0
      ad : Dvd.dvd Ordinal.omega0 a'.repr
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
    have al := split_add_lt e₁
    have aa : repr (a' + ofNat m) = repr a' + m := by
      simp only [eq_self_iff_true, ONote.repr_ofNat, ONote.repr_add]
    /-
      case mk.intro.oadd.intro
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this✝ : a0.NF
      this : a'.NF
      a00 : Ne a0.repr 0
      ad : Dvd.dvd Ordinal.omega0 a'.repr
      al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
    cases' e₂ : split' o₂ with b' k
    /-
      case mk.intro.oadd.intro.mk
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this✝ : a0.NF
      this : a'.NF
      a00 : Ne a0.repr 0
      ad : Dvd.dvd Ordinal.omega0 a'.repr
      al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
    cases' nf_repr_split' e₂ with _ r₂
    /-
      case mk.intro.oadd.intro.mk.intro
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this✝ : a0.NF
      this : a'.NF
      a00 : Ne a0.repr 0
      ad : Dvd.dvd Ordinal.omega0 a'.repr
      al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      left✝ : b'.NF
      r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
      ⊢ Eq (HPow.hPow o₁ o₂).repr (HPow.hPow o₁.repr o₂.repr)
    -/
    simp only [opow_def, opow, e₁, r₁, split_eq_scale_split' e₂, opowAux2, repr]
    /-
      case mk.intro.oadd.intro.mk.intro
      o₁ o₂ : ONote
      inst✝¹ : o₁.NF
      inst✝ : o₂.NF
      m : Nat
      a0 : ONote
      n : PNat
      a' : ONote
      e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
      N₁ : (a0.oadd n a').NF
      r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
      this✝ : a0.NF
      this : a'.NF
      a00 : Ne a0.repr 0
      ad : Dvd.dvd Ordinal.omega0 a'.repr
      al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
      aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
      b' : ONote
      k : Nat
      e₂ : Eq o₂.split' { fst := b', snd := k }
      left✝ : b'.NF
      r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑k)
      ⊢ Eq (ONote.opowAux2.match_1 (fun x => ONote) { fst := ONote.scale 1 b', snd : …
    -/
    cases' k with k
      /-
        case mk.intro.oadd.intro.mk.intro.zero
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        e₂ : Eq o₂.split' { fst := b', snd := 0 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑0)
        ⊢ Eq (ONote.opowAux2.match_1 (fun x => ONote) { fst := ONote.scale 1 b', snd : …
      -/
    · simp [r₂, opow_mul, repr_opow_aux₁ a00 al aa, add_assoc]
      /-
        🎉 no goals
      -/
    · simp? [r₂, opow_add, opow_mul, mul_assoc, add_assoc, -repr, -opow_natCast] says
        simp only [mulNat_eq_mul, repr_add, repr_scale, repr_mul, repr_ofNat, opow_add, opow_mul,
          mul_assoc, add_assoc, r₂, Nat.cast_add, Nat.cast_one, add_one_eq_succ, opow_succ]
      /-
        case mk.intro.oadd.intro.mk.intro.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑(HAdd.hAdd k 1))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (HPow.hPow (HPow.hPow Ordinal.omega0 a0. …
      -/
      simp only [repr, opow_zero, Nat.succPNat_coe, Nat.cast_one, mul_one, add_zero, opow_one]
      /-
        case mk.intro.oadd.intro.mk.intro.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑(HAdd.hAdd k 1))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (HPow.hPow (HPow.hPow Ordinal.omega0 a0. …
      -/
      rw [repr_opow_aux₁ a00 al aa, scale_opowAux]
      simp only [repr_mul, repr_scale, repr, opow_zero, Nat.succPNat_coe, Nat.cast_one, mul_one,
        add_zero, opow_one, opow_mul]
      /-
        case mk.intro.oadd.intro.mk.intro.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑(HAdd.hAdd k 1))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (HPow.hPow (HPow.hPow Ordinal.omega0 a0. …
      -/
      rw [← mul_add, ← add_assoc ((ω : Ordinal.{0}) ^ repr a0 * (n : ℕ))]
      /-
        case mk.intro.oadd.intro.mk.intro.succ
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑(HAdd.hAdd k 1))
        ⊢ Eq (HMul.hMul (HPow.hPow (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) Ordin …
      -/
      congr 1
      /-
        case mk.intro.oadd.intro.mk.intro.succ.e_a
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑(HAdd.hAdd k 1))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) ↑k) ( …
      -/
      rw [← opow_succ]
      /-
        case mk.intro.oadd.intro.mk.intro.succ.e_a
        o₁ o₂ : ONote
        inst✝¹ : o₁.NF
        inst✝ : o₂.NF
        m : Nat
        a0 : ONote
        n : PNat
        a' : ONote
        e₁ : Eq o₁.split { fst := a0.oadd n a', snd := m }
        N₁ : (a0.oadd n a').NF
        r₁ : Eq o₁.repr (HAdd.hAdd (a0.oadd n a').repr ↑m)
        this✝ : a0.NF
        this : a'.NF
        a00 : Ne a0.repr 0
        ad : Dvd.dvd Ordinal.omega0 a'.repr
        al : LT.lt (HAdd.hAdd a'.repr ↑m) (HPow.hPow Ordinal.omega0 a0.repr)
        aa : Eq (HAdd.hAdd a' ↑m).repr (HAdd.hAdd a'.repr ↑m)
        b' : ONote
        left✝ : b'.NF
        k : Nat
        e₂ : Eq o₂.split' { fst := b', snd := HAdd.hAdd k 1 }
        r₂ : Eq o₂.repr (HAdd.hAdd (HMul.hMul Ordinal.omega0 b'.repr) ↑(HAdd.hAdd k 1))
        ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (HPow.hPow Ordinal.omega0 a0.repr) ↑k) ( …
      -/
      exact (repr_opow_aux₂ _ ad a00 al _ _).2
      /-
        🎉 no goals
      -/


/-- Given an ordinal, returns:

* `inl none` for `0`
* `inl (some a)` for `a + 1`
* `inr f` for a limit ordinal `a`, where `f i` is a sequence converging to `a` -/
def fundamentalSequence : ONote → (Option ONote) ⊕ (ℕ → ONote)
  | zero => Sum.inl none
  | oadd a m b =>
    match fundamentalSequence b with
    | Sum.inr f => Sum.inr fun i => oadd a m (f i)
    | Sum.inl (some b') => Sum.inl (some (oadd a m b'))
    | Sum.inl none =>
      match fundamentalSequence a, m.natPred with
      | Sum.inl none, 0 => Sum.inl (some zero)
      | Sum.inl none, m + 1 => Sum.inl (some (oadd zero m.succPNat zero))
      | Sum.inl (some a'), 0 => Sum.inr fun i => oadd a' i.succPNat zero
      | Sum.inl (some a'), m + 1 => Sum.inr fun i => oadd a m.succPNat (oadd a' i.succPNat zero)
      | Sum.inr f, 0 => Sum.inr fun i => oadd (f i) 1 zero
      | Sum.inr f, m + 1 => Sum.inr fun i => oadd a m.succPNat (oadd (f i) 1 zero)


private theorem exists_lt_add {α} [hα : Nonempty α] {o : Ordinal} {f : α → Ordinal}
    (H : ∀ ⦃a⦄, a < o → ∃ i, a < f i) {b : Ordinal} ⦃a⦄ (h : a < b + o) : ∃ i, a < b + f i := by
  /-
    α : Sort u_1
    hα : Nonempty α
    o : Ordinal.{u_2}
    f : α → Ordinal.{u_2}
    H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
    b a : Ordinal.{u_2}
    h : LT.lt a (HAdd.hAdd b o)
    ⊢ Exists fun i => LT.lt a (HAdd.hAdd b (f i))
  -/
  cases' lt_or_le a b with h h'
    /-
      case inl
      α : Sort u_1
      hα : Nonempty α
      o : Ordinal.{u_2}
      f : α → Ordinal.{u_2}
      H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
      b a : Ordinal.{u_2}
      h✝ : LT.lt a (HAdd.hAdd b o)
      h : LT.lt a b
      ⊢ Exists fun i => LT.lt a (HAdd.hAdd b (f i))
    -/
  · obtain ⟨i⟩ := id hα
    /-
      case inl.intro
      α : Sort u_1
      hα : Nonempty α
      o : Ordinal.{u_2}
      f : α → Ordinal.{u_2}
      H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
      b a : Ordinal.{u_2}
      h✝ : LT.lt a (HAdd.hAdd b o)
      h : LT.lt a b
      i : α
      ⊢ Exists fun i => LT.lt a (HAdd.hAdd b (f i))
    -/
    exact ⟨i, h.trans_le (le_add_right _ _)⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Sort u_1
      hα : Nonempty α
      o : Ordinal.{u_2}
      f : α → Ordinal.{u_2}
      H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
      b a : Ordinal.{u_2}
      h : LT.lt a (HAdd.hAdd b o)
      h' : LE.le b a
      ⊢ Exists fun i => LT.lt a (HAdd.hAdd b (f i))
    -/
  · rw [← Ordinal.add_sub_cancel_of_le h', add_lt_add_iff_left] at h
    /-
      case inr
      α : Sort u_1
      hα : Nonempty α
      o : Ordinal.{u_2}
      f : α → Ordinal.{u_2}
      H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
      b a : Ordinal.{u_2}
      h : LT.lt (HSub.hSub a b) o
      h' : LE.le b a
      ⊢ Exists fun i => LT.lt a (HAdd.hAdd b (f i))
    -/
    refine (H h).imp fun i H => ?_
    /-
      case inr
      α : Sort u_1
      hα : Nonempty α
      o : Ordinal.{u_2}
      f : α → Ordinal.{u_2}
      H✝ : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
      b a : Ordinal.{u_2}
      h : LT.lt (HSub.hSub a b) o
      h' : LE.le b a
      i : α
      H : LT.lt (HSub.hSub a b) (f i)
      ⊢ LT.lt a (HAdd.hAdd b (f i))
    -/
    rwa [← Ordinal.add_sub_cancel_of_le h', add_lt_add_iff_left]
    /-
      🎉 no goals
    -/


private theorem exists_lt_mul_omega0' {o : Ordinal} ⦃a⦄ (h : a < o * ω) :
    ∃ i : ℕ, a < o * ↑i + o := by
  /-
    o a : Ordinal.{u_1}
    h : LT.lt a (HMul.hMul o Ordinal.omega0)
    ⊢ Exists fun i => LT.lt a (HAdd.hAdd (HMul.hMul o ↑i) o)
  -/
  obtain ⟨i, hi, h'⟩ := (lt_mul_of_limit isLimit_omega0).1 h
  /-
    case intro.intro
    o a : Ordinal.{u_1}
    h : LT.lt a (HMul.hMul o Ordinal.omega0)
    i : Ordinal.{u_1}
    hi : LT.lt i Ordinal.omega0
    h' : LT.lt a (HMul.hMul o i)
    ⊢ Exists fun i => LT.lt a (HAdd.hAdd (HMul.hMul o ↑i) o)
  -/
  obtain ⟨i, rfl⟩ := lt_omega0.1 hi
  /-
    case intro.intro.intro
    o a : Ordinal.{u_1}
    h : LT.lt a (HMul.hMul o Ordinal.omega0)
    i : Nat
    hi : LT.lt (↑i) Ordinal.omega0
    h' : LT.lt a (HMul.hMul o ↑i)
    ⊢ Exists fun i => LT.lt a (HAdd.hAdd (HMul.hMul o ↑i) o)
  -/
  exact ⟨i, h'.trans_le (le_add_right _ _)⟩
  /-
    🎉 no goals
  -/


private theorem exists_lt_omega0_opow' {α} {o b : Ordinal} (hb : 1 < b) (ho : o.IsLimit)
    {f : α → Ordinal} (H : ∀ ⦃a⦄, a < o → ∃ i, a < f i) ⦃a⦄ (h : a < b ^ o) :
        ∃ i, a < b ^ f i := by
  /-
    α : Sort u_1
    o b : Ordinal.{u_2}
    hb : LT.lt 1 b
    ho : o.IsLimit
    f : α → Ordinal.{u_2}
    H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
    a : Ordinal.{u_2}
    h : LT.lt a (HPow.hPow b o)
    ⊢ Exists fun i => LT.lt a (HPow.hPow b (f i))
  -/
  obtain ⟨d, hd, h'⟩ := (lt_opow_of_limit (zero_lt_one.trans hb).ne' ho).1 h
  /-
    case intro.intro
    α : Sort u_1
    o b : Ordinal.{u_2}
    hb : LT.lt 1 b
    ho : o.IsLimit
    f : α → Ordinal.{u_2}
    H : ∀ ⦃a : Ordinal.{u_2}⦄, LT.lt a o → Exists fun i => LT.lt a (f i)
    a : Ordinal.{u_2}
    h : LT.lt a (HPow.hPow b o)
    d : Ordinal.{u_2}
    hd : LT.lt d o
    h' : LT.lt a (HPow.hPow b d)
    ⊢ Exists fun i => LT.lt a (HPow.hPow b (f i))
  -/
  exact (H hd).imp fun i hi => h'.trans <| (opow_lt_opow_iff_right hb).2 hi
  /-
    🎉 no goals
  -/


/-- The property satisfied by `fundamentalSequence o`:

* `inl none` means `o = 0`
* `inl (some a)` means `o = succ a`
* `inr f` means `o` is a limit ordinal and `f` is a strictly increasing sequence which converges to
  `o` -/
def FundamentalSequenceProp (o : ONote) : (Option ONote) ⊕ (ℕ → ONote) → Prop
  | Sum.inl none => o = 0
  | Sum.inl (some a) => o.repr = succ a.repr ∧ (o.NF → a.NF)
  | Sum.inr f =>
    o.repr.IsLimit ∧
      (∀ i, f i < f (i + 1) ∧ f i < o ∧ (o.NF → (f i).NF)) ∧ ∀ a, a < o.repr → ∃ i, a < (f i).repr


theorem fundamentalSequenceProp_inl_none (o) :
    FundamentalSequenceProp o (Sum.inl none) ↔ o = 0 :=
  Iff.rfl


theorem fundamentalSequenceProp_inl_some (o a) :
    FundamentalSequenceProp o (Sum.inl (some a)) ↔ o.repr = succ a.repr ∧ (o.NF → a.NF) :=
  Iff.rfl


theorem fundamentalSequenceProp_inr (o f) :
    FundamentalSequenceProp o (Sum.inr f) ↔
      o.repr.IsLimit ∧
        (∀ i, f i < f (i + 1) ∧ f i < o ∧ (o.NF → (f i).NF)) ∧
        ∀ a, a < o.repr → ∃ i, a < (f i).repr :=
  Iff.rfl


theorem fundamentalSequence_has_prop (o) : FundamentalSequenceProp o (fundamentalSequence o) := by
  /-
    o : ONote
    ⊢ o.FundamentalSequenceProp o.fundamentalSequence
  -/
  induction' o with a m b iha ihb; · exact rfl
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case oadd
    a : ONote
    m : PNat
    b : ONote
    iha : a.FundamentalSequenceProp a.fundamentalSequence
    ihb : b.FundamentalSequenceProp b.fundamentalSequence
    ⊢ (a.oadd m b).FundamentalSequenceProp (a.oadd m b).fundamentalSequence
  -/
  rw [fundamentalSequence]
  /-
    case oadd
    a : ONote
    m : PNat
    b : ONote
    iha : a.FundamentalSequenceProp a.fundamentalSequence
    ihb : b.FundamentalSequenceProp b.fundamentalSequence
    ⊢ (a.oadd m b).FundamentalSequenceProp (ONote.fundamentalSequence.match_2 (fun …
  -/
  rcases e : b.fundamentalSequence with (⟨_ | b'⟩ | f) <;>
    /-
      case oadd.inl.none
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      ihb : b.FundamentalSequenceProp b.fundamentalSequence
      e : Eq b.fundamentalSequence (Sum.inl Option.none)
      ⊢ (a.oadd m b).FundamentalSequenceProp (ONote.fundamentalSequence.match_2 (fun …
    -/
    simp only [FundamentalSequenceProp] <;>
    /-
      case oadd.inl.none
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      ihb : b.FundamentalSequenceProp b.fundamentalSequence
      e : Eq b.fundamentalSequence (Sum.inl Option.none)
      ⊢ ONote.FundamentalSequenceProp.match_1 (fun x => Prop) (ONote.fundamentalSequ …
    -/
    rw [e, FundamentalSequenceProp] at ihb
    /-
      case oadd.inl.none
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      ihb : Eq b 0
      e : Eq b.fundamentalSequence (Sum.inl Option.none)
      ⊢ ONote.FundamentalSequenceProp.match_1 (fun x => Prop) (ONote.fundamentalSequ …
    -/
  · rcases e : a.fundamentalSequence with (⟨_ | a'⟩ | f) <;> cases' e' : m.natPred with m' <;>
      /-
        case oadd.inl.none.inl.none.zero
        a : ONote
        m : PNat
        b : ONote
        iha : a.FundamentalSequenceProp a.fundamentalSequence
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        e : Eq a.fundamentalSequence (Sum.inl Option.none)
        e' : Eq m.natPred 0
        ⊢ ONote.FundamentalSequenceProp.match_1 (fun x => Prop) (ONote.fundamentalSequ …
      -/
      simp only [FundamentalSequenceProp] <;>
      /-
        case oadd.inl.none.inl.none.zero
        a : ONote
        m : PNat
        b : ONote
        iha : a.FundamentalSequenceProp a.fundamentalSequence
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        e : Eq a.fundamentalSequence (Sum.inl Option.none)
        e' : Eq m.natPred 0
        ⊢ And (Eq (a.oadd m b).repr (Order.succ ONote.zero.repr)) ((a.oadd m b).NF → O …
      -/
      rw [e, FundamentalSequenceProp] at iha <;>
      (try rw [show m = 1 by
            have := PNat.natPred_add_one m; rw [e'] at this; exact PNat.coe_inj.1 this.symm]) <;>
      (try rw [show m = (m' + 1).succPNat by
              rw [← e', ← PNat.coe_inj, Nat.succPNat_coe, ← Nat.add_one, PNat.natPred_add_one]]) <;>
      simp only [repr, iha, ihb, opow_lt_opow_iff_right one_lt_omega0, add_lt_add_iff_left,
        add_zero, eq_self_iff_true, lt_add_iff_pos_right, lt_def, mul_one, Nat.cast_zero,
        Nat.cast_succ, Nat.succPNat_coe, opow_succ, opow_zero, mul_add_one, PNat.one_coe, succ_zero,
        _root_.zero_add, zero_def]
      /-
        case oadd.inl.none.inl.none.zero
        a : ONote
        m : PNat
        b : ONote
        iha : Eq a 0
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        e : Eq a.fundamentalSequence (Sum.inl Option.none)
        e' : Eq m.natPred 0
        ⊢ And True ((ONote.oadd 0 1 0).NF → ONote.NF 0)
      -/
    · decide
      /-
        🎉 no goals
      -/
      /-
        case oadd.inl.none.inl.none.succ
        a : ONote
        m : PNat
        b : ONote
        iha : Eq a 0
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        e : Eq a.fundamentalSequence (Sum.inl Option.none)
        m' : Nat
        e' : Eq m.natPred (HAdd.hAdd m' 1)
        ⊢ And (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul 1 ↑m') 1) 1) (Order.succ (HAdd.hAdd …
      -/
    · exact ⟨rfl, inferInstance⟩
      /-
        🎉 no goals
      -/
      /-
        case oadd.inl.none.inl.some.zero
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        a' : ONote
        iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
        e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
        e' : Eq m.natPred 0
        ⊢ And (HMul.hMul (HPow.hPow Ordinal.omega0 a'.repr) Ordinal.omega0).IsLimit (A …
      -/
    · have := opow_pos (repr a') omega0_pos
      refine
        ⟨isLimit_mul this isLimit_omega0, fun i =>
          ⟨this, ?_, fun H => @NF.oadd_zero _ _ (iha.2 H.fst)⟩, exists_lt_mul_omega0'⟩
      /-
        case oadd.inl.none.inl.some.zero
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        a' : ONote
        iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
        e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
        e' : Eq m.natPred 0
        this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
        i : Nat
        ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 a'.repr) ↑i) (HPow.hPo …
      -/
      rw [← mul_succ, ← natCast_succ, Ordinal.mul_lt_mul_iff_left this]
      /-
        case oadd.inl.none.inl.some.zero
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        a' : ONote
        iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
        e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
        e' : Eq m.natPred 0
        this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
        i : Nat
        ⊢ LT.lt (↑i.succ) Ordinal.omega0
      -/
      apply nat_lt_omega0
      /-
        🎉 no goals
      -/
      /-
        case oadd.inl.none.inl.some.succ
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        a' : ONote
        iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
        e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
        m' : Nat
        e' : Eq m.natPred (HAdd.hAdd m' 1)
        ⊢ And (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow Ordinal.omega0 a' …
      -/
    · have := opow_pos (repr a') omega0_pos
      refine
        ⟨isLimit_add _ (isLimit_mul this isLimit_omega0), fun i => ⟨this, ?_, ?_⟩,
          exists_lt_add exists_lt_mul_omega0'⟩
        /-
          case oadd.inl.none.inl.some.succ.refine_1
          a : ONote
          m : PNat
          b : ONote
          ihb : Eq b 0
          e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
          a' : ONote
          iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
          e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
          m' : Nat
          e' : Eq m.natPred (HAdd.hAdd m' 1)
          this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
          i : Nat
          ⊢ LT.lt (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 a'.repr) ↑i) (HPow.hPo …
        -/
      · rw [← mul_succ, ← natCast_succ, Ordinal.mul_lt_mul_iff_left this]
        /-
          case oadd.inl.none.inl.some.succ.refine_1
          a : ONote
          m : PNat
          b : ONote
          ihb : Eq b 0
          e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
          a' : ONote
          iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
          e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
          m' : Nat
          e' : Eq m.natPred (HAdd.hAdd m' 1)
          this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
          i : Nat
          ⊢ LT.lt (↑i.succ) Ordinal.omega0
        -/
        apply nat_lt_omega0
        /-
          🎉 no goals
        -/
        /-
          case oadd.inl.none.inl.some.succ.refine_2
          a : ONote
          m : PNat
          b : ONote
          ihb : Eq b 0
          e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
          a' : ONote
          iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
          e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
          m' : Nat
          e' : Eq m.natPred (HAdd.hAdd m' 1)
          this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
          i : Nat
          ⊢ (a.oadd (HAdd.hAdd m' 1).succPNat 0).NF → (a.oadd m'.succPNat (a'.oadd i.suc …
        -/
      · refine fun H => H.fst.oadd _ (NF.below_of_lt' ?_ (@NF.oadd_zero _ _ (iha.2 H.fst)))
        /-
          case oadd.inl.none.inl.some.succ.refine_2
          a : ONote
          m : PNat
          b : ONote
          ihb : Eq b 0
          e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
          a' : ONote
          iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
          e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
          m' : Nat
          e' : Eq m.natPred (HAdd.hAdd m' 1)
          this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
          i : Nat
          H : (a.oadd (HAdd.hAdd m' 1).succPNat 0).NF
          ⊢ LT.lt (a'.oadd i.succPNat 0).repr (HPow.hPow Ordinal.omega0 a.repr)
        -/
        rw [repr, ← zero_def, repr, add_zero, iha.1, opow_succ, Ordinal.mul_lt_mul_iff_left this]
        /-
          case oadd.inl.none.inl.some.succ.refine_2
          a : ONote
          m : PNat
          b : ONote
          ihb : Eq b 0
          e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
          a' : ONote
          iha : And (Eq a.repr (Order.succ a'.repr)) (a.NF → a'.NF)
          e : Eq a.fundamentalSequence (Sum.inl (Option.some a'))
          m' : Nat
          e' : Eq m.natPred (HAdd.hAdd m' 1)
          this : LT.lt 0 (HPow.hPow Ordinal.omega0 a'.repr)
          i : Nat
          H : (a.oadd (HAdd.hAdd m' 1).succPNat 0).NF
          ⊢ LT.lt (↑↑i.succPNat) Ordinal.omega0
        -/
        apply nat_lt_omega0
        /-
          🎉 no goals
        -/
      /-
        case oadd.inl.none.inr.zero
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        f : Nat → ONote
        iha : And a.repr.IsLimit (And (∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1 …
        e : Eq a.fundamentalSequence (Sum.inr f)
        e' : Eq m.natPred 0
        ⊢ And (HPow.hPow Ordinal.omega0 a.repr).IsLimit (And (∀ (i : Nat), And (LT.lt  …
      -/
    · rcases iha with ⟨h1, h2, h3⟩
      refine ⟨isLimit_opow one_lt_omega0 h1, fun i => ?_,
        exists_lt_omega0_opow' one_lt_omega0 h1 h3⟩
      /-
        case oadd.inl.none.inr.zero.intro.intro
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        f : Nat → ONote
        e : Eq a.fundamentalSequence (Sum.inr f)
        e' : Eq m.natPred 0
        h1 : a.repr.IsLimit
        h2 : ∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1))) (And (LT.lt (f i) a) ( …
        h3 : ∀ (a_1 : Ordinal.{0}), LT.lt a_1 a.repr → Exists fun i => LT.lt a_1 (f i) …
        i : Nat
        ⊢ And (LT.lt (f i).repr (f (HAdd.hAdd i 1)).repr) (And (LT.lt (f i).repr a.rep …
      -/
      obtain ⟨h4, h5, h6⟩ := h2 i
      /-
        case oadd.inl.none.inr.zero.intro.intro.intro.intro
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        f : Nat → ONote
        e : Eq a.fundamentalSequence (Sum.inr f)
        e' : Eq m.natPred 0
        h1 : a.repr.IsLimit
        h2 : ∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1))) (And (LT.lt (f i) a) ( …
        h3 : ∀ (a_1 : Ordinal.{0}), LT.lt a_1 a.repr → Exists fun i => LT.lt a_1 (f i) …
        i : Nat
        h4 : LT.lt (f i) (f (HAdd.hAdd i 1))
        h5 : LT.lt (f i) a
        h6 : a.NF → (f i).NF
        ⊢ And (LT.lt (f i).repr (f (HAdd.hAdd i 1)).repr) (And (LT.lt (f i).repr a.rep …
      -/
      exact ⟨h4, h5, fun H => @NF.oadd_zero _ _ (h6 H.fst)⟩
      /-
        🎉 no goals
      -/
      /-
        case oadd.inl.none.inr.succ
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        f : Nat → ONote
        iha : And a.repr.IsLimit (And (∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1 …
        e : Eq a.fundamentalSequence (Sum.inr f)
        m' : Nat
        e' : Eq m.natPred (HAdd.hAdd m' 1)
        ⊢ And (HAdd.hAdd (HAdd.hAdd (HMul.hMul (HPow.hPow Ordinal.omega0 a.repr) ↑m')  …
      -/
    · rcases iha with ⟨h1, h2, h3⟩
      refine
        ⟨isLimit_add _ (isLimit_opow one_lt_omega0 h1), fun i => ?_,
          exists_lt_add (exists_lt_omega0_opow' one_lt_omega0 h1 h3)⟩
      /-
        case oadd.inl.none.inr.succ.intro.intro
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        f : Nat → ONote
        e : Eq a.fundamentalSequence (Sum.inr f)
        m' : Nat
        e' : Eq m.natPred (HAdd.hAdd m' 1)
        h1 : a.repr.IsLimit
        h2 : ∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1))) (And (LT.lt (f i) a) ( …
        h3 : ∀ (a_1 : Ordinal.{0}), LT.lt a_1 a.repr → Exists fun i => LT.lt a_1 (f i) …
        i : Nat
        ⊢ And (LT.lt (f i).repr (f (HAdd.hAdd i 1)).repr) (And (LT.lt (f i).repr a.rep …
      -/
      obtain ⟨h4, h5, h6⟩ := h2 i
      /-
        case oadd.inl.none.inr.succ.intro.intro.intro.intro
        a : ONote
        m : PNat
        b : ONote
        ihb : Eq b 0
        e✝ : Eq b.fundamentalSequence (Sum.inl Option.none)
        f : Nat → ONote
        e : Eq a.fundamentalSequence (Sum.inr f)
        m' : Nat
        e' : Eq m.natPred (HAdd.hAdd m' 1)
        h1 : a.repr.IsLimit
        h2 : ∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1))) (And (LT.lt (f i) a) ( …
        h3 : ∀ (a_1 : Ordinal.{0}), LT.lt a_1 a.repr → Exists fun i => LT.lt a_1 (f i) …
        i : Nat
        h4 : LT.lt (f i) (f (HAdd.hAdd i 1))
        h5 : LT.lt (f i) a
        h6 : a.NF → (f i).NF
        ⊢ And (LT.lt (f i).repr (f (HAdd.hAdd i 1)).repr) (And (LT.lt (f i).repr a.rep …
      -/
      refine ⟨h4, h5, fun H => H.fst.oadd _ (NF.below_of_lt' ?_ (@NF.oadd_zero _ _ (h6 H.fst)))⟩
      rwa [repr, ← zero_def, repr, add_zero, PNat.one_coe, Nat.cast_one, mul_one,
        opow_lt_opow_iff_right one_lt_omega0]
  · refine ⟨by
      rw [repr, ihb.1, add_succ, repr], fun H => H.fst.oadd _ (NF.below_of_lt' ?_ (ihb.2 H.snd))⟩
    /-
      case oadd.inl.some
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      b' : ONote
      ihb : And (Eq b.repr (Order.succ b'.repr)) (b.NF → b'.NF)
      e : Eq b.fundamentalSequence (Sum.inl (Option.some b'))
      H : (a.oadd m b).NF
      ⊢ LT.lt b'.repr (HPow.hPow Ordinal.omega0 a.repr)
    -/
    have := H.snd'.repr_lt
    /-
      case oadd.inl.some
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      b' : ONote
      ihb : And (Eq b.repr (Order.succ b'.repr)) (b.NF → b'.NF)
      e : Eq b.fundamentalSequence (Sum.inl (Option.some b'))
      H : (a.oadd m b).NF
      this : LT.lt b.repr (HPow.hPow Ordinal.omega0 a.repr)
      ⊢ LT.lt b'.repr (HPow.hPow Ordinal.omega0 a.repr)
    -/
    rw [ihb.1] at this
    /-
      case oadd.inl.some
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      b' : ONote
      ihb : And (Eq b.repr (Order.succ b'.repr)) (b.NF → b'.NF)
      e : Eq b.fundamentalSequence (Sum.inl (Option.some b'))
      H : (a.oadd m b).NF
      this : LT.lt (Order.succ b'.repr) (HPow.hPow Ordinal.omega0 a.repr)
      ⊢ LT.lt b'.repr (HPow.hPow Ordinal.omega0 a.repr)
    -/
    exact (lt_succ _).trans this
    /-
      🎉 no goals
    -/
    /-
      case oadd.inr
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      f : Nat → ONote
      ihb : And b.repr.IsLimit (And (∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1 …
      e : Eq b.fundamentalSequence (Sum.inr f)
      ⊢ And (a.oadd m b).repr.IsLimit (And (∀ (i : Nat), And (LT.lt (a.oadd m (f i)) …
    -/
  · rcases ihb with ⟨h1, h2, h3⟩
    /-
      case oadd.inr.intro.intro
      a : ONote
      m : PNat
      b : ONote
      iha : a.FundamentalSequenceProp a.fundamentalSequence
      f : Nat → ONote
      e : Eq b.fundamentalSequence (Sum.inr f)
      h1 : b.repr.IsLimit
      h2 : ∀ (i : Nat), And (LT.lt (f i) (f (HAdd.hAdd i 1))) (And (LT.lt (f i) b) ( …
      h3 : ∀ (a : Ordinal.{0}), LT.lt a b.repr → Exists fun i => LT.lt a (f i).repr
      ⊢ And (a.oadd m b).repr.IsLimit (And (∀ (i : Nat), And (LT.lt (a.oadd m (f i)) …
    -/
    simp only [repr]
    exact
      ⟨Ordinal.isLimit_add _ h1, fun i =>
        ⟨oadd_lt_oadd_3 (h2 i).1, oadd_lt_oadd_3 (h2 i).2.1, fun H =>
          H.fst.oadd _ (NF.below_of_lt' (lt_trans (h2 i).2.1 H.snd'.repr_lt) ((h2 i).2.2 H.snd))⟩,
        exists_lt_add h3⟩


/-- The fast growing hierarchy for ordinal notations `< ε₀`. This is a sequence of functions `ℕ → ℕ`
indexed by ordinals, with the definition:

* `f_0(n) = n + 1`
* `f_(α + 1)(n) = f_α^[n](n)`
* `f_α(n) = f_(α[n])(n)` where `α` is a limit ordinal and `α[i]` is the fundamental sequence
  converging to `α` -/
def fastGrowing : ONote → ℕ → ℕ
  | o =>
    match fundamentalSequence o, fundamentalSequence_has_prop o with
    | Sum.inl none, _ => Nat.succ
    | Sum.inl (some a), h =>
                         /-
                           x✝ : ONote
                           o : ONote := x✝
                           a : ONote
                           h : o.FundamentalSequenceProp (Sum.inl (Option.some a))
                           ⊢ LT.lt a o
                         -/
      have : a < o := by rw [lt_def, h.1]; apply lt_succ
                                           /-
                                             🎉 no goals
                                           -/
      fun i => (fastGrowing a)^[i] i
    | Sum.inr f, h => fun i =>
      have : f i < o := (h.2.1 i).2.1
      fastGrowing (f i) i
  termination_by o => o

-- Porting note: the linter bug should be fixed.

@[nolint unusedHavesSuffices]
theorem fastGrowing_def {o : ONote} {x} (e : fundamentalSequence o = x) :
    fastGrowing o =
      match
        (motive := (x : Option ONote ⊕ (ℕ → ONote)) → FundamentalSequenceProp o x → ℕ → ℕ)
        x, e ▸ fundamentalSequence_has_prop o with
      | Sum.inl none, _ => Nat.succ
      | Sum.inl (some a), _ =>
        fun i => (fastGrowing a)^[i] i
      | Sum.inr f, _ => fun i =>
        fastGrowing (f i) i := by
  /-
    o : ONote
    x : Sum (Option ONote) (Nat → ONote)
    e : Eq o.fundamentalSequence x
    ⊢ Eq o.fastGrowing (ONote.fastGrowing.match_1 o (fun x a => Nat → Nat) x ⋯ (fu …
  -/
  subst x
  /-
    o : ONote
    ⊢ Eq o.fastGrowing (ONote.fastGrowing.match_1 o (fun x a => Nat → Nat) o.funda …
  -/
  rw [fastGrowing]
  /-
    🎉 no goals
  -/


theorem fastGrowing_zero' (o : ONote) (h : fundamentalSequence o = Sum.inl none) :
    fastGrowing o = Nat.succ := by
  /-
    o : ONote
    h : Eq o.fundamentalSequence (Sum.inl Option.none)
    ⊢ Eq o.fastGrowing Nat.succ
  -/
  rw [fastGrowing_def h]
  /-
    🎉 no goals
  -/


theorem fastGrowing_succ (o) {a} (h : fundamentalSequence o = Sum.inl (some a)) :
    fastGrowing o = fun i => (fastGrowing a)^[i] i := by
  /-
    o a : ONote
    h : Eq o.fundamentalSequence (Sum.inl (Option.some a))
    ⊢ Eq o.fastGrowing fun i => Nat.iterate a.fastGrowing i i
  -/
  rw [fastGrowing_def h]
  /-
    🎉 no goals
  -/


theorem fastGrowing_limit (o) {f} (h : fundamentalSequence o = Sum.inr f) :
    fastGrowing o = fun i => fastGrowing (f i) i := by
  /-
    o : ONote
    f : Nat → ONote
    h : Eq o.fundamentalSequence (Sum.inr f)
    ⊢ Eq o.fastGrowing fun i => (f i).fastGrowing i
  -/
  rw [fastGrowing_def h]
  /-
    🎉 no goals
  -/


@[simp]
theorem fastGrowing_zero : fastGrowing 0 = Nat.succ :=
  fastGrowing_zero' _ rfl


@[simp]
theorem fastGrowing_one : fastGrowing 1 = fun n => 2 * n := by
  /-
    ⊢ Eq (ONote.fastGrowing 1) fun n => HMul.hMul 2 n
  -/
  rw [@fastGrowing_succ 1 0 rfl]; funext i; rw [two_mul, fastGrowing_zero]
  /-
    case h
    i : Nat
    ⊢ Eq (Nat.iterate Nat.succ i i) (HAdd.hAdd i i)
  -/
  suffices ∀ a b, Nat.succ^[a] b = b + a from this _ _
  /-
    case h
    i : Nat
    ⊢ ∀ (a b : Nat), Eq (Nat.iterate Nat.succ a b) (HAdd.hAdd b a)
  -/
                             /-
                               🎉 no goals
                             -/
  intro a b; induction a <;> simp [*, Function.iterate_succ', Nat.add_assoc, -Function.iterate_succ]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem fastGrowing_two : fastGrowing 2 = fun n => (2 ^ n) * n := by
  /-
    ⊢ Eq (ONote.fastGrowing 2) fun n => HMul.hMul (HPow.hPow 2 n) n
  -/
  rw [@fastGrowing_succ 2 1 rfl]; funext i; rw [fastGrowing_one]
  /-
    case h
    i : Nat
    ⊢ Eq (Nat.iterate (fun n => HMul.hMul 2 n) i i) (HMul.hMul (HPow.hPow 2 i) i)
  -/
  suffices ∀ a b, (fun n : ℕ => 2 * n)^[a] b = (2 ^ a) * b from this _ _
  /-
    case h
    i : Nat
    ⊢ ∀ (a b : Nat), Eq (Nat.iterate (fun n => HMul.hMul 2 n) a b) (HMul.hMul (HPo …
  -/
  intro a b; induction a <;>
    /-
      case h.zero
      i b : Nat
      ⊢ Eq (Nat.iterate (fun n => HMul.hMul 2 n) 0 b) (HMul.hMul (HPow.hPow 2 0) b)
    -/
    /-
      🎉 no goals
    -/
    simp [*, Function.iterate_succ, pow_succ, mul_assoc, -Function.iterate_succ]
    /-
      🎉 no goals
    -/


/-- We can extend the fast growing hierarchy one more step to `ε₀` itself, using `ω ^ (ω ^ (⋯ ^ ω))`
as the fundamental sequence converging to `ε₀` (which is not an `ONote`). Extending the fast
growing hierarchy beyond this requires a definition of fundamental sequence for larger ordinals. -/
def fastGrowingε₀ (i : ℕ) : ℕ :=
  fastGrowing ((fun a => a.oadd 1 0)^[i] 0) i


                                                       /-
                                                         ⊢ Eq (ONote.fastGrowingε₀ 0) 1
                                                       -/
theorem fastGrowingε₀_zero : fastGrowingε₀ 0 = 1 := by simp [fastGrowingε₀]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem fastGrowingε₀_one : fastGrowingε₀ 1 = 2 := by
  /-
    ⊢ Eq (ONote.fastGrowingε₀ 1) 2
  -/
  simp [fastGrowingε₀, show oadd 0 1 0 = 1 from rfl]
  /-
    🎉 no goals
  -/


theorem fastGrowingε₀_two : fastGrowingε₀ 2 = 2048 := by
  norm_num [fastGrowingε₀, show oadd 0 1 0 = 1 from rfl, @fastGrowing_limit (oadd 1 1 0) _ rfl,
    show oadd 0 (2 : Nat).succPNat 0 = 3 from rfl, @fastGrowing_succ 3 2 rfl]


/-- The type of normal ordinal notations.

It would have been nicer to define this right in the inductive type, but `NF o` requires `repr`
which requires `ONote`, so all these things would have to be defined at once, which messes up the VM
representation. -/
def NONote :=
  { o : ONote // o.NF }


                                    /-
                                      ⊢ DecidableEq NONote
                                    -/
instance : DecidableEq NONote := by unfold NONote; infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


instance NF (o : NONote) : NF o.1 :=
  o.2


/-- Construct a `NONote` from an ordinal notation (and infer normality) -/
def mk (o : ONote) [h : ONote.NF o] : NONote :=
  ⟨o, h⟩


/-- The ordinal represented by an ordinal notation.

This function is noncomputable because ordinal arithmetic is noncomputable. In computational
applications `NONote` can be used exclusively without reference to `Ordinal`, but this function
allows for correctness results to be stated. -/
noncomputable def repr (o : NONote) : Ordinal :=
  o.1.repr


instance : ToString NONote :=
  ⟨fun x => x.1.toString⟩


instance : Repr NONote :=
  ⟨fun x prec => x.1.repr' prec⟩


instance : Preorder NONote where
  le x y := repr x ≤ repr y
  lt x y := repr x < repr y
  le_refl _ := @le_refl Ordinal _ _
  le_trans _ _ _ := @le_trans Ordinal _ _ _ _
  lt_iff_le_not_le _ _ := @lt_iff_le_not_le Ordinal _ _ _


instance : Zero NONote :=
  ⟨⟨0, NF.zero⟩⟩


instance : Inhabited NONote :=
  ⟨0⟩


theorem lt_wf : @WellFounded NONote (· < ·) :=
  InvImage.wf repr Ordinal.lt_wf


instance : WellFoundedLT NONote :=
  ⟨lt_wf⟩


instance : WellFoundedRelation NONote :=
  ⟨(· < ·), lt_wf⟩


/-- Convert a natural number to an ordinal notation -/
def ofNat (n : ℕ) : NONote :=
  ⟨ONote.ofNat n, ⟨⟨_, nfBelow_ofNat _⟩⟩⟩


/-- Compare ordinal notations -/
def cmp (a b : NONote) : Ordering :=
  ONote.cmp a.1 b.1


theorem cmp_compares : ∀ a b : NONote, (cmp a b).Compares a b
  | ⟨a, ha⟩, ⟨b, hb⟩ => by
    /-
      a : ONote
      ha : a.NF
      b : ONote
      hb : b.NF
      ⊢ (NONote.cmp ⟨a, ha⟩ ⟨b, hb⟩).Compares ⟨a, ha⟩ ⟨b, hb⟩
    -/
    dsimp [cmp]
    /-
      a : ONote
      ha : a.NF
      b : ONote
      hb : b.NF
      ⊢ (a.cmp b).Compares ⟨a, ha⟩ ⟨b, hb⟩
    -/
    have := ONote.cmp_compares a b
    /-
      a : ONote
      ha : a.NF
      b : ONote
      hb : b.NF
      this : (a.cmp b).Compares a b
      ⊢ (a.cmp b).Compares ⟨a, ha⟩ ⟨b, hb⟩
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    cases h : ONote.cmp a b <;> simp only [h] at this <;> try exact this
                                                          /-
                                                            🎉 no goals
                                                          -/
    /-
      case eq
      a : ONote
      ha : a.NF
      b : ONote
      hb : b.NF
      h : Eq (a.cmp b) Ordering.eq
      this : Ordering.eq.Compares a b
      ⊢ Ordering.eq.Compares ⟨a, ha⟩ ⟨b, hb⟩
    -/
    exact Subtype.mk_eq_mk.2 this
    /-
      🎉 no goals
    -/


instance : LinearOrder NONote :=
  linearOrderOfCompares cmp cmp_compares


/-- Asserts that `repr a < ω ^ repr b`. Used in `NONote.recOn`. -/
def below (a b : NONote) : Prop :=
  NFBelow a.1 (repr b)


/-- The `oadd` pseudo-constructor for `NONote` -/
def oadd (e : NONote) (n : ℕ+) (a : NONote) (h : below a e) : NONote :=
  ⟨_, NF.oadd e.2 n h⟩


/-- This is a recursor-like theorem for `NONote` suggesting an inductive definition, which can't
actually be defined this way due to conflicting dependencies. -/
@[elab_as_elim]
def recOn {C : NONote → Sort*} (o : NONote) (H0 : C 0)
    (H1 : ∀ e n a h, C e → C a → C (oadd e n a h)) : C o := by
  /-
    C : NONote → Sort u_1
    o : NONote
    H0 : C 0
    H1 : (e : NONote) → (n : PNat) → (a : NONote) → (h : a.below e) → C e → C a →  …
    ⊢ C o
  -/
  cases' o with o h; induction' o with e n a IHe IHa
    /-
      case mk.zero
      C : NONote → Sort u_1
      H0 : C 0
      H1 : (e : NONote) → (n : PNat) → (a : NONote) → (h : a.below e) → C e → C a →  …
      h : ONote.zero.NF
      ⊢ C ⟨ONote.zero, h⟩
    -/
  · exact H0
    /-
      🎉 no goals
    -/
    /-
      case mk.oadd
      C : NONote → Sort u_1
      H0 : C 0
      H1 : (e : NONote) → (n : PNat) → (a : NONote) → (h : a.below e) → C e → C a →  …
      e : ONote
      n : PNat
      a : ONote
      IHe : (h : e.NF) → C ⟨e, h⟩
      IHa : (h : a.NF) → C ⟨a, h⟩
      h : (e.oadd n a).NF
      ⊢ C ⟨e.oadd n a, h⟩
    -/
  · exact H1 ⟨e, h.fst⟩ n ⟨a, h.snd⟩ h.snd' (IHe _) (IHa _)
    /-
      🎉 no goals
    -/


/-- Addition of ordinal notations -/
instance : Add NONote :=
  ⟨fun x y => mk (x.1 + y.1)⟩


theorem repr_add (a b) : repr (a + b) = repr a + repr b :=
  ONote.repr_add a.1 b.1


/-- Subtraction of ordinal notations -/
instance : Sub NONote :=
  ⟨fun x y => mk (x.1 - y.1)⟩


theorem repr_sub (a b) : repr (a - b) = repr a - repr b :=
  ONote.repr_sub a.1 b.1


/-- Multiplication of ordinal notations -/
instance : Mul NONote :=
  ⟨fun x y => mk (x.1 * y.1)⟩


theorem repr_mul (a b) : repr (a * b) = repr a * repr b :=
  ONote.repr_mul a.1 b.1


/-- Exponentiation of ordinal notations -/
def opow (x y : NONote) :=
  mk (x.1 ^ y.1)


theorem repr_opow (a b) : repr (opow a b) = repr a ^ repr b :=
  ONote.repr_opow a.1 b.1


