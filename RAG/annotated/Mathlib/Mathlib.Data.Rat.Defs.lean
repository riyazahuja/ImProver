theorem pos (a : ℚ) : 0 < a.den := Nat.pos_of_ne_zero a.den_nz


lemma mk'_num_den (q : ℚ) : mk' q.num q.den q.den_nz q.reduced = q := rfl


@[simp]
theorem ofInt_eq_cast (n : ℤ) : ofInt n = Int.cast n :=
  rfl

-- TODO: Replace `Rat.ofNat_num`/`Rat.ofNat_den` in Batteries

@[simp] lemma num_ofNat (n : ℕ) : num ofNat(n) = ofNat(n) := rfl

@[simp] lemma den_ofNat (n : ℕ) : den ofNat(n) = 1 := rfl


@[simp, norm_cast] lemma num_natCast (n : ℕ) : num n = n := rfl


@[simp, norm_cast] lemma den_natCast (n : ℕ) : den n = 1 := rfl

-- TODO: Replace `intCast_num`/`intCast_den` the names in Batteries

@[simp, norm_cast] lemma num_intCast (n : ℤ) : (n : ℚ).num = n := rfl


@[simp, norm_cast] lemma den_intCast (n : ℤ) : (n : ℚ).den = 1 := rfl


@[deprecated (since := "2024-04-29")] alias coe_int_num := num_intCast

@[deprecated (since := "2024-04-29")] alias coe_int_den := den_intCast


lemma intCast_injective : Injective (Int.cast : ℤ → ℚ) := fun _ _ ↦ congr_arg num

lemma natCast_injective : Injective (Nat.cast : ℕ → ℚ) :=
  intCast_injective.comp fun _ _ ↦ Int.natCast_inj.1

-- We want to use these lemmas earlier than the lemmas simp can prove them with

@[simp, nolint simpNF, norm_cast] lemma natCast_inj {m n : ℕ} : (m : ℚ) = n ↔ m = n :=
  natCast_injective.eq_iff

@[simp, nolint simpNF, norm_cast] lemma intCast_eq_zero {n : ℤ} : (n : ℚ) = 0 ↔ n = 0 := intCast_inj

@[simp, nolint simpNF, norm_cast] lemma natCast_eq_zero {n : ℕ} : (n : ℚ) = 0 ↔ n = 0 := natCast_inj

@[simp, nolint simpNF, norm_cast] lemma intCast_eq_one {n : ℤ} : (n : ℚ) = 1 ↔ n = 1 := intCast_inj

@[simp, nolint simpNF, norm_cast] lemma natCast_eq_one {n : ℕ} : (n : ℚ) = 1 ↔ n = 1 := natCast_inj

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO Should this be namespaced?


lemma mkRat_eq_divInt (n d) : mkRat n d = n /. d := rfl


                                                                   /-
                                                                     d : Nat
                                                                     h : Ne d 0
                                                                     w : (Int.natAbs 0).Coprime d
                                                                     ⊢ Eq { num := 0, den := d, den_nz := h, reduced := w } 0
                                                                   -/
@[simp] lemma mk'_zero (d) (h : d ≠ 0) (w) : mk' 0 d h w = 0 := by congr; simp_all
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
lemma num_eq_zero {q : ℚ} : q.num = 0 ↔ q = 0 := by
  /-
    q : Rat
    ⊢ Iff (Eq q.num 0) (Eq q 0)
  -/
  induction q
  /-
    case mk'
    num✝ : Int
    den✝ : Nat
    den_nz✝ : Ne den✝ 0
    reduced✝ : num✝.natAbs.Coprime den✝
    ⊢ Iff (Eq { num := num✝, den := den✝, den_nz := den_nz✝, reduced := reduced✝ } …
  -/
  constructor
    /-
      case mk'.mp
      num✝ : Int
      den✝ : Nat
      den_nz✝ : Ne den✝ 0
      reduced✝ : num✝.natAbs.Coprime den✝
      ⊢ Eq { num := num✝, den := den✝, den_nz := den_nz✝, reduced := reduced✝ }.num  …
    -/
  · rintro rfl
    /-
      case mk'.mp
      den✝ : Nat
      den_nz✝ : Ne den✝ 0
      reduced✝ : (Int.natAbs 0).Coprime den✝
      ⊢ Eq { num := 0, den := den✝, den_nz := den_nz✝, reduced := reduced✝ } 0
    -/
    exact mk'_zero _ _ _
    /-
      🎉 no goals
    -/
    /-
      case mk'.mpr
      num✝ : Int
      den✝ : Nat
      den_nz✝ : Ne den✝ 0
      reduced✝ : num✝.natAbs.Coprime den✝
      ⊢ Eq { num := num✝, den := den✝, den_nz := den_nz✝, reduced := reduced✝ } 0 →  …
    -/
  · exact congr_arg num
    /-
      🎉 no goals
    -/


lemma num_ne_zero {q : ℚ} : q.num ≠ 0 ↔ q ≠ 0 := num_eq_zero.not


@[simp] lemma den_ne_zero (q : ℚ) : q.den ≠ 0 := q.den_pos.ne'


@[simp] lemma num_nonneg : 0 ≤ q.num ↔ 0 ≤ q := by
  /-
    q : Rat
    ⊢ Iff (LE.le 0 q.num) (LE.le 0 q)
  -/
  simp [Int.le_iff_lt_or_eq, instLE, Rat.blt, Int.not_lt]; tauto
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem divInt_eq_zero {a b : ℤ} (b0 : b ≠ 0) : a /. b = 0 ↔ a = 0 := by
  /-
    a b : Int
    b0 : Ne b 0
    ⊢ Iff (Eq (Rat.divInt a b) 0) (Eq a 0)
  -/
  rw [← zero_divInt b, divInt_eq_iff b0 b0, Int.zero_mul, Int.mul_eq_zero, or_iff_left b0]
  /-
    🎉 no goals
  -/


theorem divInt_ne_zero {a b : ℤ} (b0 : b ≠ 0) : a /. b ≠ 0 ↔ a ≠ 0 :=
  (divInt_eq_zero b0).not

-- Porting note: this can move to Batteries

theorem normalize_eq_mk' (n : Int) (d : Nat) (h : d ≠ 0) (c : Nat.gcd (Int.natAbs n) d = 1) :
    normalize n d h = mk' n d h c := (mk_eq_normalize ..).symm

-- TODO: Rename `mkRat_num_den` in Batteries

@[simp] alias mkRat_num_den' := mkRat_self

-- TODO: Rename `Rat.divInt_self` to `Rat.num_divInt_den` in Batteries

lemma num_divInt_den (q : ℚ) : q.num /. q.den = q := divInt_self _


lemma mk'_eq_divInt {n d h c} : (⟨n, d, h, c⟩ : ℚ) = n /. d := (num_divInt_den _).symm


theorem intCast_eq_divInt (z : ℤ) : (z : ℚ) = z /. 1 := mk'_eq_divInt

-- TODO: Rename `divInt_self` in Batteries to `num_divInt_den`

@[simp] lemma divInt_self' {n : ℤ} (hn : n ≠ 0) : n /. n = 1 := by
  /-
    n : Int
    hn : Ne n 0
    ⊢ Eq (Rat.divInt n n) 1
  -/
  simpa using divInt_mul_right (n := 1) (d := 1) hn
  /-
    🎉 no goals
  -/


/-- Define a (dependent) function or prove `∀ r : ℚ, p r` by dealing with rational
numbers of the form `n /. d` with `0 < d` and coprime `n`, `d`. -/
@[elab_as_elim]
def numDenCasesOn.{u} {C : ℚ → Sort u} :
    ∀ (a : ℚ) (_ : ∀ n d, 0 < d → (Int.natAbs n).Coprime d → C (n /. d)), C a
                          /-
                            q : Rat
                            C : Rat → Sort u
                            n : Int
                            d : Nat
                            h : Ne d 0
                            c : n.natAbs.Coprime d
                            H : (n : Int) → (d : Nat) → LT.lt 0 d → n.natAbs.Coprime d → C (Rat.divInt n ↑d)
                            ⊢ C { num := n, den := d, den_nz := h, reduced := c }
                          -/
  | ⟨n, d, h, c⟩, H => by rw [mk'_eq_divInt]; exact H n d (Nat.pos_of_ne_zero h) c
                                              /-
                                                🎉 no goals
                                              -/


/-- Define a (dependent) function or prove `∀ r : ℚ, p r` by dealing with rational
numbers of the form `n /. d` with `d ≠ 0`. -/
@[elab_as_elim]
def numDenCasesOn'.{u} {C : ℚ → Sort u} (a : ℚ) (H : ∀ (n : ℤ) (d : ℕ), d ≠ 0 → C (n /. d)) :
    C a :=
  numDenCasesOn a fun n d h _ => H n d h.ne'


/-- Define a (dependent) function or prove `∀ r : ℚ, p r` by dealing with rational
numbers of the form `mk' n d` with `d ≠ 0`. -/
@[elab_as_elim]
def numDenCasesOn''.{u} {C : ℚ → Sort u} (a : ℚ)
    (H : ∀ (n : ℤ) (d : ℕ) (nz red), C (mk' n d nz red)) : C a :=
                                    /-
                                      q : Rat
                                      C : Rat → Sort u
                                      a : Rat
                                      H : (n : Int) → (d : Nat) → (nz : Ne d 0) → (red : n.natAbs.Coprime d) → C { n …
                                      n : Int
                                      d : Nat
                                      h : LT.lt 0 d
                                      h' : n.natAbs.Coprime d
                                      ⊢ C (Rat.divInt n ↑d)
                                    -/
  numDenCasesOn a fun n d h h' ↦ by rw [← mk_eq_divInt _ _ h.ne' h']; exact H n d h.ne' _
                                                                      /-
                                                                        🎉 no goals
                                                                      -/

-- Porting note: there's already an instance for `Add ℚ` is in Batteries.


theorem lift_binop_eq (f : ℚ → ℚ → ℚ) (f₁ : ℤ → ℤ → ℤ → ℤ → ℤ) (f₂ : ℤ → ℤ → ℤ → ℤ → ℤ)
    (fv :
      ∀ {n₁ d₁ h₁ c₁ n₂ d₂ h₂ c₂},
        f ⟨n₁, d₁, h₁, c₁⟩ ⟨n₂, d₂, h₂, c₂⟩ = f₁ n₁ d₁ n₂ d₂ /. f₂ n₁ d₁ n₂ d₂)
    (f0 : ∀ {n₁ d₁ n₂ d₂}, d₁ ≠ 0 → d₂ ≠ 0 → f₂ n₁ d₁ n₂ d₂ ≠ 0) (a b c d : ℤ)
    (b0 : b ≠ 0) (d0 : d ≠ 0)
    (H :
      ∀ {n₁ d₁ n₂ d₂}, a * d₁ = n₁ * b → c * d₂ = n₂ * d →
        f₁ n₁ d₁ n₂ d₂ * f₂ a b c d = f₁ a b c d * f₂ n₁ d₁ n₂ d₂) :
    f (a /. b) (c /. d) = f₁ a b c d /. f₂ a b c d := by
  /-
    f : Rat → Rat → Rat
    f₁ f₂ : Int → Int → Int → Int → Int
    fv : ∀ {n₁ : Int} {d₁ : Nat} {h₁ : Ne d₁ 0} {c₁ : n₁.natAbs.Coprime d₁} {n₂ :  …
    f0 : ∀ {n₁ d₁ n₂ d₂ : Int}, Ne d₁ 0 → Ne d₂ 0 → Ne (f₂ n₁ d₁ n₂ d₂) 0
    a b c d : Int
    b0 : Ne b 0
    d0 : Ne d 0
    H : ∀ {n₁ d₁ n₂ d₂ : Int}, Eq (HMul.hMul a d₁) (HMul.hMul n₁ b) → Eq (HMul.hMu …
    ⊢ Eq (f (Rat.divInt a b) (Rat.divInt c d)) (Rat.divInt (f₁ a b c d) (f₂ a b c  …
  -/
  generalize ha : a /. b = x; cases' x with n₁ d₁ h₁ c₁; rw [mk'_eq_divInt] at ha
  /-
    case mk'
    f : Rat → Rat → Rat
    f₁ f₂ : Int → Int → Int → Int → Int
    fv : ∀ {n₁ : Int} {d₁ : Nat} {h₁ : Ne d₁ 0} {c₁ : n₁.natAbs.Coprime d₁} {n₂ :  …
    f0 : ∀ {n₁ d₁ n₂ d₂ : Int}, Ne d₁ 0 → Ne d₂ 0 → Ne (f₂ n₁ d₁ n₂ d₂) 0
    a b c d : Int
    b0 : Ne b 0
    d0 : Ne d 0
    H : ∀ {n₁ d₁ n₂ d₂ : Int}, Eq (HMul.hMul a d₁) (HMul.hMul n₁ b) → Eq (HMul.hMu …
    n₁ : Int
    d₁ : Nat
    h₁ : Ne d₁ 0
    c₁ : n₁.natAbs.Coprime d₁
    ha : Eq (Rat.divInt a b) (Rat.divInt n₁ ↑d₁)
    ⊢ Eq (f { num := n₁, den := d₁, den_nz := h₁, reduced := c₁ } (Rat.divInt c d) …
  -/
  generalize hc : c /. d = x; cases' x with n₂ d₂ h₂ c₂; rw [mk'_eq_divInt] at hc
  /-
    case mk'.mk'
    f : Rat → Rat → Rat
    f₁ f₂ : Int → Int → Int → Int → Int
    fv : ∀ {n₁ : Int} {d₁ : Nat} {h₁ : Ne d₁ 0} {c₁ : n₁.natAbs.Coprime d₁} {n₂ :  …
    f0 : ∀ {n₁ d₁ n₂ d₂ : Int}, Ne d₁ 0 → Ne d₂ 0 → Ne (f₂ n₁ d₁ n₂ d₂) 0
    a b c d : Int
    b0 : Ne b 0
    d0 : Ne d 0
    H : ∀ {n₁ d₁ n₂ d₂ : Int}, Eq (HMul.hMul a d₁) (HMul.hMul n₁ b) → Eq (HMul.hMu …
    n₁ : Int
    d₁ : Nat
    h₁ : Ne d₁ 0
    c₁ : n₁.natAbs.Coprime d₁
    ha : Eq (Rat.divInt a b) (Rat.divInt n₁ ↑d₁)
    n₂ : Int
    d₂ : Nat
    h₂ : Ne d₂ 0
    c₂ : n₂.natAbs.Coprime d₂
    hc : Eq (Rat.divInt c d) (Rat.divInt n₂ ↑d₂)
    ⊢ Eq (f { num := n₁, den := d₁, den_nz := h₁, reduced := c₁ } { num := n₂, den …
  -/
  rw [fv]
  /-
    case mk'.mk'
    f : Rat → Rat → Rat
    f₁ f₂ : Int → Int → Int → Int → Int
    fv : ∀ {n₁ : Int} {d₁ : Nat} {h₁ : Ne d₁ 0} {c₁ : n₁.natAbs.Coprime d₁} {n₂ :  …
    f0 : ∀ {n₁ d₁ n₂ d₂ : Int}, Ne d₁ 0 → Ne d₂ 0 → Ne (f₂ n₁ d₁ n₂ d₂) 0
    a b c d : Int
    b0 : Ne b 0
    d0 : Ne d 0
    H : ∀ {n₁ d₁ n₂ d₂ : Int}, Eq (HMul.hMul a d₁) (HMul.hMul n₁ b) → Eq (HMul.hMu …
    n₁ : Int
    d₁ : Nat
    h₁ : Ne d₁ 0
    c₁ : n₁.natAbs.Coprime d₁
    ha : Eq (Rat.divInt a b) (Rat.divInt n₁ ↑d₁)
    n₂ : Int
    d₂ : Nat
    h₂ : Ne d₂ 0
    c₂ : n₂.natAbs.Coprime d₂
    hc : Eq (Rat.divInt c d) (Rat.divInt n₂ ↑d₂)
    ⊢ Eq (Rat.divInt (f₁ n₁ (↑d₁) n₂ ↑d₂) (f₂ n₁ (↑d₁) n₂ ↑d₂)) (Rat.divInt (f₁ a  …
  -/
  have d₁0 := Int.ofNat_ne_zero.2 h₁
  /-
    case mk'.mk'
    f : Rat → Rat → Rat
    f₁ f₂ : Int → Int → Int → Int → Int
    fv : ∀ {n₁ : Int} {d₁ : Nat} {h₁ : Ne d₁ 0} {c₁ : n₁.natAbs.Coprime d₁} {n₂ :  …
    f0 : ∀ {n₁ d₁ n₂ d₂ : Int}, Ne d₁ 0 → Ne d₂ 0 → Ne (f₂ n₁ d₁ n₂ d₂) 0
    a b c d : Int
    b0 : Ne b 0
    d0 : Ne d 0
    H : ∀ {n₁ d₁ n₂ d₂ : Int}, Eq (HMul.hMul a d₁) (HMul.hMul n₁ b) → Eq (HMul.hMu …
    n₁ : Int
    d₁ : Nat
    h₁ : Ne d₁ 0
    c₁ : n₁.natAbs.Coprime d₁
    ha : Eq (Rat.divInt a b) (Rat.divInt n₁ ↑d₁)
    n₂ : Int
    d₂ : Nat
    h₂ : Ne d₂ 0
    c₂ : n₂.natAbs.Coprime d₂
    hc : Eq (Rat.divInt c d) (Rat.divInt n₂ ↑d₂)
    d₁0 : Ne (↑d₁) 0
    ⊢ Eq (Rat.divInt (f₁ n₁ (↑d₁) n₂ ↑d₂) (f₂ n₁ (↑d₁) n₂ ↑d₂)) (Rat.divInt (f₁ a  …
  -/
  have d₂0 := Int.ofNat_ne_zero.2 h₂
  exact (divInt_eq_iff (f0 d₁0 d₂0) (f0 b0 d0)).2
    (H ((divInt_eq_iff b0 d₁0).1 ha) ((divInt_eq_iff d0 d₂0).1 hc))


@[deprecated divInt_add_divInt (since := "2024-03-18")]
theorem add_def'' {a b c d : ℤ} (b0 : b ≠ 0) (d0 : d ≠ 0) :
    a /. b + c /. d = (a * d + c * b) /. (b * d) := divInt_add_divInt _ _ b0 d0


                                                   /-
                                                     q : Rat
                                                     ⊢ Eq (Neg.neg q) (Rat.divInt (Neg.neg q.num) ↑q.den)
                                                   -/
lemma neg_def (q : ℚ) : -q = -q.num /. q.den := by rw [← neg_divInt, num_divInt_den]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] lemma divInt_neg (n d : ℤ) : n /. -d = -n /. d := divInt_neg' ..


@[deprecated (since := "2024-03-18")] alias divInt_neg_den := divInt_neg


@[deprecated divInt_sub_divInt (since := "2024-03-18")]
lemma sub_def'' {a b c d : ℤ} (b0 : b ≠ 0) (d0 : d ≠ 0) :
    a /. b - c /. d = (a * d - c * b) /. (b * d) := divInt_sub_divInt _ _ b0 d0


@[simp]
lemma divInt_mul_divInt' (n₁ d₁ n₂ d₂ : ℤ) : (n₁ /. d₁) * (n₂ /. d₂) = (n₁ * n₂) /. (d₁ * d₂) := by
  /-
    n₁ d₁ n₂ d₂ : Int
    ⊢ Eq (HMul.hMul (Rat.divInt n₁ d₁) (Rat.divInt n₂ d₂)) (Rat.divInt (HMul.hMul  …
  -/
  obtain rfl | h₁ := eq_or_ne d₁ 0
    /-
      case inl
      n₁ n₂ d₂ : Int
      ⊢ Eq (HMul.hMul (Rat.divInt n₁ 0) (Rat.divInt n₂ d₂)) (Rat.divInt (HMul.hMul n …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    n₁ d₁ n₂ d₂ : Int
    h₁ : Ne d₁ 0
    ⊢ Eq (HMul.hMul (Rat.divInt n₁ d₁) (Rat.divInt n₂ d₂)) (Rat.divInt (HMul.hMul  …
  -/
  obtain rfl | h₂ := eq_or_ne d₂ 0
    /-
      case inr.inl
      n₁ d₁ n₂ : Int
      h₁ : Ne d₁ 0
      ⊢ Eq (HMul.hMul (Rat.divInt n₁ d₁) (Rat.divInt n₂ 0)) (Rat.divInt (HMul.hMul n …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    n₁ d₁ n₂ d₂ : Int
    h₁ : Ne d₁ 0
    h₂ : Ne d₂ 0
    ⊢ Eq (HMul.hMul (Rat.divInt n₁ d₁) (Rat.divInt n₂ d₂)) (Rat.divInt (HMul.hMul  …
  -/
  exact divInt_mul_divInt _ _ h₁ h₂
  /-
    🎉 no goals
  -/


lemma mk'_mul_mk' (n₁ n₂ : ℤ) (d₁ d₂ : ℕ) (hd₁ hd₂ hnd₁ hnd₂) (h₁₂ : n₁.natAbs.Coprime d₂)
    (h₂₁ : n₂.natAbs.Coprime d₁) :
    mk' n₁ d₁ hd₁ hnd₁ * mk' n₂ d₂ hd₂ hnd₂ = mk' (n₁ * n₂) (d₁ * d₂) (Nat.mul_ne_zero hd₁ hd₂) (by
      /-
        q : Rat
        n₁ n₂ : Int
        d₁ d₂ : Nat
        hd₁ : Ne d₁ 0
        hd₂ : Ne d₂ 0
        hnd₁ : n₁.natAbs.Coprime d₁
        hnd₂ : n₂.natAbs.Coprime d₂
        h₁₂ : n₁.natAbs.Coprime d₂
        h₂₁ : n₂.natAbs.Coprime d₁
        ⊢ (HMul.hMul n₁ n₂).natAbs.Coprime (HMul.hMul d₁ d₂)
      -/
      rw [Int.natAbs_mul]; exact (hnd₁.mul h₂₁).mul_right (h₁₂.mul hnd₂)) := by
                           /-
                             🎉 no goals
                           -/
  /-
    n₁ n₂ : Int
    d₁ d₂ : Nat
    hd₁ : Ne d₁ 0
    hd₂ : Ne d₂ 0
    hnd₁ : n₁.natAbs.Coprime d₁
    hnd₂ : n₂.natAbs.Coprime d₂
    h₁₂ : n₁.natAbs.Coprime d₂
    h₂₁ : n₂.natAbs.Coprime d₁
    ⊢ Eq (HMul.hMul { num := n₁, den := d₁, den_nz := hd₁, reduced := hnd₁ } { num …
  -/
  rw [mul_def]; dsimp; simp [mk_eq_normalize]
                       /-
                         🎉 no goals
                       -/


lemma mul_eq_mkRat (q r : ℚ) : q * r = mkRat (q.num * r.num) (q.den * r.den) := by
  /-
    q r : Rat
    ⊢ Eq (HMul.hMul q r) (mkRat (HMul.hMul q.num r.num) (HMul.hMul q.den r.den))
  -/
  rw [mul_def, normalize_eq_mkRat]
  /-
    🎉 no goals
  -/

-- TODO: Rename `divInt_eq_iff` in Batteries to `divInt_eq_divInt`

alias divInt_eq_divInt := divInt_eq_iff


@[deprecated (since := "2024-04-29")] alias mul_num_den := mul_eq_mkRat


instance instPowNat : Pow ℚ ℕ where
                                       /-
                                         q✝ q : Rat
                                         n : Nat
                                         ⊢ Ne (HPow.hPow q.den n) 0
                                       -/
  pow q n := ⟨q.num ^ n, q.den ^ n, by simp [Nat.pow_eq_zero], by
                                       /-
                                         🎉 no goals
                                       -/
    /-
      q✝ q : Rat
      n : Nat
      ⊢ (HPow.hPow q.num n).natAbs.Coprime (HPow.hPow q.den n)
    -/
    rw [Int.natAbs_pow]; exact q.reduced.pow _ _⟩
                         /-
                           🎉 no goals
                         -/


lemma pow_def (q : ℚ) (n : ℕ) :
    q ^ n = ⟨q.num ^ n, q.den ^ n,
         /-
           q✝ q : Rat
           n : Nat
           ⊢ Ne (HPow.hPow q.den n) 0
         -/
      by simp [Nat.pow_eq_zero],
         /-
           🎉 no goals
         -/
         /-
           q✝ q : Rat
           n : Nat
           ⊢ (HPow.hPow q.num n).natAbs.Coprime (HPow.hPow q.den n)
         -/
      by rw [Int.natAbs_pow]; exact q.reduced.pow _ _⟩ := rfl
                              /-
                                🎉 no goals
                              -/


lemma pow_eq_mkRat (q : ℚ) (n : ℕ) : q ^ n = mkRat (q.num ^ n) (q.den ^ n) := by
  /-
    q : Rat
    n : Nat
    ⊢ Eq (HPow.hPow q n) (mkRat (HPow.hPow q.num n) (HPow.hPow q.den n))
  -/
  rw [pow_def, mk_eq_mkRat]
  /-
    🎉 no goals
  -/


lemma pow_eq_divInt (q : ℚ) (n : ℕ) : q ^ n = q.num ^ n /. q.den ^ n := by
  /-
    q : Rat
    n : Nat
    ⊢ Eq (HPow.hPow q n) (Rat.divInt (HPow.hPow q.num n) (HPow.hPow (↑q.den) n))
  -/
  rw [pow_def, mk_eq_divInt, Int.natCast_pow]
  /-
    🎉 no goals
  -/


@[simp] lemma num_pow (q : ℚ) (n : ℕ) : (q ^ n).num = q.num ^ n := rfl

@[simp] lemma den_pow (q : ℚ) (n : ℕ) : (q ^ n).den = q.den ^ n := rfl


@[simp] lemma mk'_pow (num : ℤ) (den : ℕ) (hd hdn) (n : ℕ) :
    mk' num den hd hdn ^ n = mk' (num ^ n) (den ^ n)
          /-
            q : Rat
            num : Int
            den : Nat
            hd : Ne den 0
            hdn : num.natAbs.Coprime den
            n : Nat
            ⊢ Ne (HPow.hPow den n) 0
          -/
          /-
            🎉 no goals
          -/
      (by simp [Nat.pow_eq_zero, hd]) (by rw [Int.natAbs_pow]; exact hdn.pow _ _) := rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance : Inv ℚ :=
  ⟨Rat.inv⟩


@[simp] lemma inv_divInt' (a b : ℤ) : (a /. b)⁻¹ = b /. a := inv_divInt ..


@[simp] lemma inv_mkRat (a : ℤ) (b : ℕ) : (mkRat a b)⁻¹ = b /. a := by
  /-
    a : Int
    b : Nat
    ⊢ Eq (Inv.inv (mkRat a b)) (Rat.divInt (↑b) a)
  -/
  rw [mkRat_eq_divInt, inv_divInt']
  /-
    🎉 no goals
  -/


                                                    /-
                                                      q : Rat
                                                      ⊢ Eq (Inv.inv q) (Rat.divInt (↑q.den) q.num)
                                                    -/
lemma inv_def' (q : ℚ) : q⁻¹ = q.den /. q.num := by rw [← inv_divInt', num_divInt_den]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp] lemma divInt_div_divInt (n₁ d₁ n₂ d₂) :
    (n₁ /. d₁) / (n₂ /. d₂) = (n₁ * d₂) /. (d₁ * n₂) := by
  /-
    n₁ d₁ n₂ d₂ : Int
    ⊢ Eq (HDiv.hDiv (Rat.divInt n₁ d₁) (Rat.divInt n₂ d₂)) (Rat.divInt (HMul.hMul  …
  -/
  rw [div_def, inv_divInt, divInt_mul_divInt']
  /-
    🎉 no goals
  -/


lemma div_def' (q r : ℚ) : q / r = (q.num * r.den) /. (q.den * r.num) := by
  /-
    q r : Rat
    ⊢ Eq (HDiv.hDiv q r) (Rat.divInt (HMul.hMul q.num ↑r.den) (HMul.hMul (↑q.den)  …
  -/
  rw [← divInt_div_divInt, num_divInt_den, num_divInt_den]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-15")] alias div_num_den := div_def'


                                           /-
                                             a : Rat
                                             ⊢ Eq (HAdd.hAdd a 0) a
                                           -/
protected lemma add_zero : a + 0 = a := by simp [add_def, normalize_eq_mkRat]
                                           /-
                                             🎉 no goals
                                           -/


                                           /-
                                             a : Rat
                                             ⊢ Eq (HAdd.hAdd 0 a) a
                                           -/
protected lemma zero_add : 0 + a = a := by simp [add_def, normalize_eq_mkRat]
                                           /-
                                             🎉 no goals
                                           -/


protected lemma add_comm : a + b = b + a := by
  /-
    a b : Rat
    ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
  -/
  simp [add_def, Int.add_comm, Int.mul_comm, Nat.mul_comm]
  /-
    🎉 no goals
  -/


protected theorem add_assoc : a + b + c = a + (b + c) :=
  numDenCasesOn' a fun n₁ d₁ h₁ ↦ numDenCasesOn' b fun n₂ d₂ h₂ ↦ numDenCasesOn' c fun n₃ d₃ h₃ ↦ by
    simp only [ne_eq, Int.natCast_eq_zero, h₁, not_false_eq_true, h₂, divInt_add_divInt,
      Int.mul_eq_zero, or_self, h₃]
    /-
      a b c : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      n₃ : Int
      d₃ : Nat
      h₃ : Ne d₃ 0
      ⊢ Eq (Rat.divInt (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul n₁ ↑d₂) (HMul.hMu …
    -/
    rw [Int.mul_assoc, Int.add_mul, Int.add_mul, Int.mul_assoc, Int.add_assoc]
    /-
      a b c : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      n₃ : Int
      d₃ : Nat
      h₃ : Ne d₃ 0
      ⊢ Eq (Rat.divInt (HAdd.hAdd (HMul.hMul n₁ (HMul.hMul ↑d₂ ↑d₃)) (HAdd.hAdd (HMu …
    -/
    congr 2
    /-
      case e_a.e_a
      a b c : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      n₃ : Int
      d₃ : Nat
      h₃ : Ne d₃ 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul n₂ ↑d₁) ↑d₃) (HMul.hMul n₃ (HMul.hMul ↑d …
    -/
    ac_rfl
    /-
      🎉 no goals
    -/


protected lemma neg_add_cancel : -a + a = 0 := by
  /-
    a : Rat
    ⊢ Eq (HAdd.hAdd (Neg.neg a) a) 0
  -/
  simp [add_def, normalize_eq_mkRat, Int.neg_mul, Int.add_comm, ← Int.sub_eq_add_neg]
  /-
    🎉 no goals
  -/


@[deprecated zero_divInt (since := "2024-03-18")]
lemma divInt_zero_one : 0 /. 1 = 0 := zero_divInt _


                                                    /-
                                                      n : Int
                                                      ⊢ Eq (Rat.divInt n 1) ↑n
                                                    -/
@[simp] lemma divInt_one (n : ℤ) : n /. 1 = n := by simp [divInt, mkRat, normalize]
                                                    /-
                                                      🎉 no goals
                                                    -/

                                                      /-
                                                        n : Int
                                                        ⊢ Eq (mkRat n 1) ↑n
                                                      -/
@[simp] lemma mkRat_one (n : ℤ) : mkRat n 1 = n := by simp [mkRat_eq_divInt]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                        /-
                                          ⊢ Eq (Rat.divInt 1 1) 1
                                        -/
lemma divInt_one_one : 1 /. 1 = 1 := by rw [divInt_one, intCast_one]
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated divInt_one (since := "2024-03-18")]
                                              /-
                                                ⊢ Eq (Rat.divInt (-1) 1) (-1)
                                              -/
lemma divInt_neg_one_one : -1 /. 1 = -1 := by rw [divInt_one, intCast_neg, intCast_one]
                                              /-
                                                🎉 no goals
                                              -/


protected theorem mul_assoc : a * b * c = a * (b * c) :=
  numDenCasesOn' a fun n₁ d₁ h₁ =>
    numDenCasesOn' b fun n₂ d₂ h₂ =>
      numDenCasesOn' c fun n₃ d₃ h₃ => by
        /-
          a b c : Rat
          n₁ : Int
          d₁ : Nat
          h₁ : Ne d₁ 0
          n₂ : Int
          d₂ : Nat
          h₂ : Ne d₂ 0
          n₃ : Int
          d₃ : Nat
          h₃ : Ne d₃ 0
          ⊢ Eq (HMul.hMul (HMul.hMul (Rat.divInt n₁ ↑d₁) (Rat.divInt n₂ ↑d₂)) (Rat.divIn …
        -/
        simp [h₁, h₂, h₃, Int.mul_comm, Nat.mul_assoc, Int.mul_left_comm]
        /-
          🎉 no goals
        -/


protected theorem add_mul : (a + b) * c = a * c + b * c :=
  numDenCasesOn' a fun n₁ d₁ h₁ ↦ numDenCasesOn' b fun n₂ d₂ h₂ ↦ numDenCasesOn' c fun n₃ d₃ h₃ ↦ by
    simp only [ne_eq, Int.natCast_eq_zero, h₁, not_false_eq_true, h₂, divInt_add_divInt,
      Int.mul_eq_zero, or_self, h₃, divInt_mul_divInt]
    /-
      a b c : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      n₃ : Int
      d₃ : Nat
      h₃ : Ne d₃ 0
      ⊢ Eq (Rat.divInt (HMul.hMul (HAdd.hAdd (HMul.hMul n₁ ↑d₂) (HMul.hMul n₂ ↑d₁))  …
    -/
    rw [← divInt_mul_right (Int.natCast_ne_zero.2 h₃), Int.add_mul, Int.add_mul]
    /-
      a b c : Rat
      n₁ : Int
      d₁ : Nat
      h₁ : Ne d₁ 0
      n₂ : Int
      d₂ : Nat
      h₂ : Ne d₂ 0
      n₃ : Int
      d₃ : Nat
      h₃ : Ne d₃ 0
      ⊢ Eq (Rat.divInt (HAdd.hAdd (HMul.hMul (HMul.hMul (HMul.hMul n₁ ↑d₂) n₃) ↑d₃)  …
    -/
    ac_rfl
    /-
      🎉 no goals
    -/


protected theorem mul_add : a * (b + c) = a * b + a * c := by
  /-
    a b c : Rat
    ⊢ Eq (HMul.hMul a (HAdd.hAdd b c)) (HAdd.hAdd (HMul.hMul a b) (HMul.hMul a c))
  -/
  rw [Rat.mul_comm, Rat.add_mul, Rat.mul_comm, Rat.mul_comm c a]
  /-
    🎉 no goals
  -/


protected theorem zero_ne_one : 0 ≠ (1 : ℚ) := by
  /-
    ⊢ Ne 0 1
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  rw [ne_comm, ← divInt_one_one, divInt_ne_zero] <;> omega
                                                     /-
                                                       🎉 no goals
                                                     -/


attribute [simp] mkRat_eq_zero


protected theorem mul_inv_cancel : a ≠ 0 → a * a⁻¹ = 1 :=
  numDenCasesOn' a fun n d hd hn ↦ by
    /-
      a : Rat
      n : Int
      d : Nat
      hd : Ne d 0
      hn : Ne (Rat.divInt n ↑d) 0
      ⊢ Eq (HMul.hMul (Rat.divInt n ↑d) (Inv.inv (Rat.divInt n ↑d))) 1
    -/
    simp only [divInt_ofNat, ne_eq, hd, not_false_eq_true, mkRat_eq_zero] at hn
    /-
      a : Rat
      n : Int
      d : Nat
      hd : Ne d 0
      hn : Not (Eq n 0)
      ⊢ Eq (HMul.hMul (Rat.divInt n ↑d) (Inv.inv (Rat.divInt n ↑d))) 1
    -/
    simp [-divInt_ofNat, mkRat_eq_divInt, Int.mul_comm, Int.mul_ne_zero hn (Int.ofNat_ne_zero.2 hd)]
    /-
      🎉 no goals
    -/


protected theorem inv_mul_cancel (h : a ≠ 0) : a⁻¹ * a = 1 :=
  Eq.trans (Rat.mul_comm _ _) (Rat.mul_inv_cancel _ h)

-- Porting note: we already have a `DecidableEq ℚ`.

-- Extra instances to short-circuit type class resolution
-- TODO(Mario): this instance slows down Mathlib.Data.Real.Basic

                                                                     /-
                                                                       q a b c : Rat
                                                                       ⊢ Ne 1 0
                                                                     -/
instance nontrivial : Nontrivial ℚ where exists_pair_ne := ⟨1, 0, by decide⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance addCommGroup : AddCommGroup ℚ where
  zero := 0
  add := (· + ·)
  neg := Neg.neg
  zero_add := Rat.zero_add
  add_zero := Rat.add_zero
  add_comm := Rat.add_comm
  add_assoc := Rat.add_assoc
  neg_add_cancel := Rat.neg_add_cancel
  sub_eq_add_neg := Rat.sub_eq_add_neg
  nsmul := nsmulRec
  zsmul := zsmulRec


                                     /-
                                       q a b c : Rat
                                       ⊢ AddGroup Rat
                                     -/
instance addGroup : AddGroup ℚ := by infer_instance
                                     /-
                                       🎉 no goals
                                     -/


                                               /-
                                                 q a b c : Rat
                                                 ⊢ AddCommMonoid Rat
                                               -/
instance addCommMonoid : AddCommMonoid ℚ := by infer_instance
                                               /-
                                                 🎉 no goals
                                               -/


                                       /-
                                         q a b c : Rat
                                         ⊢ AddMonoid Rat
                                       -/
instance addMonoid : AddMonoid ℚ := by infer_instance
                                       /-
                                         🎉 no goals
                                       -/


                                                                 /-
                                                                   q a b c : Rat
                                                                   ⊢ AddLeftCancelSemigroup Rat
                                                                 -/
instance addLeftCancelSemigroup : AddLeftCancelSemigroup ℚ := by infer_instance
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                   /-
                                                                     q a b c : Rat
                                                                     ⊢ AddRightCancelSemigroup Rat
                                                                   -/
instance addRightCancelSemigroup : AddRightCancelSemigroup ℚ := by infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


                                                     /-
                                                       q a b c : Rat
                                                       ⊢ AddCommSemigroup Rat
                                                     -/
instance addCommSemigroup : AddCommSemigroup ℚ := by infer_instance
                                                     /-
                                                       🎉 no goals
                                                     -/


                                             /-
                                               q a b c : Rat
                                               ⊢ AddSemigroup Rat
                                             -/
instance addSemigroup : AddSemigroup ℚ := by infer_instance
                                             /-
                                               🎉 no goals
                                             -/


instance commMonoid : CommMonoid ℚ where
  one := 1
  mul := (· * ·)
  mul_one := Rat.mul_one
  one_mul := Rat.one_mul
  mul_comm := Rat.mul_comm
  mul_assoc := Rat.mul_assoc
  npow n q := q ^ n
                  /-
                    q a b c : Rat
                    ⊢ ∀ (x : Rat), Eq ((fun n q => HPow.hPow q n) 0 x) 1
                  -/
                                            /-
                                              🎉 no goals
                                            -/
  npow_zero := by intros; apply Rat.ext <;> simp [Int.pow_zero]
                                            /-
                                              🎉 no goals
                                            -/
  npow_succ n q := by
    /-
      q✝ a b c : Rat
      n : Nat
      q : Rat
      ⊢ Eq ((fun n q => HPow.hPow q n) (HAdd.hAdd n 1) q) (HMul.hMul ((fun n q => HP …
    -/
    dsimp
    /-
      q✝ a b c : Rat
      n : Nat
      q : Rat
      ⊢ Eq (HPow.hPow q (HAdd.hAdd n 1)) (HMul.hMul (HPow.hPow q n) q)
    -/
    rw [← q.mk'_num_den, mk'_pow, mk'_mul_mk']
      /-
        q✝ a b c : Rat
        n : Nat
        q : Rat
        ⊢ Eq { num := HPow.hPow q.num (HAdd.hAdd n 1), den := HPow.hPow q.den (HAdd.hA …
      -/
    · congr
      /-
        🎉 no goals
      -/
      /-
        case h₁₂
        q✝ a b c : Rat
        n : Nat
        q : Rat
        ⊢ (HPow.hPow { num := q.num, den := q.den, den_nz := ⋯, reduced := ⋯ } n).num. …
      -/
    · rw [mk'_pow, Int.natAbs_pow]
      /-
        case h₁₂
        q✝ a b c : Rat
        n : Nat
        q : Rat
        ⊢ (HPow.hPow q.num.natAbs n).Coprime q.den
      -/
      exact q.reduced.pow_left _
      /-
        🎉 no goals
      -/
      /-
        case h₂₁
        q✝ a b c : Rat
        n : Nat
        q : Rat
        ⊢ q.num.natAbs.Coprime (HPow.hPow { num := q.num, den := q.den, den_nz := ⋯, r …
      -/
    · rw [mk'_pow]
      /-
        case h₂₁
        q✝ a b c : Rat
        n : Nat
        q : Rat
        ⊢ q.num.natAbs.Coprime { num := HPow.hPow q.num n, den := HPow.hPow q.den n, d …
      -/
      exact q.reduced.pow_right _
      /-
        🎉 no goals
      -/


                                 /-
                                   q a b c : Rat
                                   ⊢ Monoid Rat
                                 -/
instance monoid : Monoid ℚ := by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


                                               /-
                                                 q a b c : Rat
                                                 ⊢ CommSemigroup Rat
                                               -/
instance commSemigroup : CommSemigroup ℚ := by infer_instance
                                               /-
                                                 🎉 no goals
                                               -/


                                       /-
                                         q a b c : Rat
                                         ⊢ Semigroup Rat
                                       -/
instance semigroup : Semigroup ℚ := by infer_instance
                                       /-
                                         🎉 no goals
                                       -/


theorem eq_iff_mul_eq_mul {p q : ℚ} : p = q ↔ p.num * q.den = q.num * p.den := by
  conv =>
    lhs
    rw [← num_divInt_den p, ← num_divInt_den q]
  /-
    p q : Rat
    ⊢ Iff (Eq (Rat.divInt p.num ↑p.den) (Rat.divInt q.num ↑q.den)) (Eq (HMul.hMul  …
  -/
  apply Rat.divInt_eq_iff <;>
      /-
        case z₁
        p q : Rat
        ⊢ Ne (↑p.den) 0
      -/
      /-
        case z₁
        p q : Rat
        ⊢ Not (Eq p.den 0)
      -/
      /-
        🎉 no goals
      -/
      /-
        case z₂
        p q : Rat
        ⊢ Not (Eq q.den 0)
      -/
      apply den_nz
      /-
        🎉 no goals
      -/


@[simp]
theorem den_neg_eq_den (q : ℚ) : (-q).den = q.den :=
  rfl


@[simp]
theorem num_neg_eq_neg_num (q : ℚ) : (-q).num = -q.num :=
  rfl

-- Not `@[simp]` as `num_ofNat` is stronger.

theorem num_zero : Rat.num 0 = 0 :=
  rfl

-- Not `@[simp]` as `den_ofNat` is stronger.

theorem den_zero : Rat.den 0 = 1 :=
  rfl


                                                              /-
                                                                q : Rat
                                                                hq : Eq q.num 0
                                                                ⊢ Eq q 0
                                                              -/
lemma zero_of_num_zero {q : ℚ} (hq : q.num = 0) : q = 0 := by simpa [hq] using q.num_divInt_den.symm
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem zero_iff_num_zero {q : ℚ} : q = 0 ↔ q.num = 0 :=
               /-
                 q : Rat
                 x✝ : Eq q 0
                 ⊢ Eq q.num 0
               -/
  ⟨fun _ => by simp [*], zero_of_num_zero⟩
               /-
                 🎉 no goals
               -/

-- `Not `@[simp]` as `num_ofNat` is stronger.

theorem num_one : (1 : ℚ).num = 1 :=
  rfl


@[simp]
theorem den_one : (1 : ℚ).den = 1 :=
  rfl


theorem mk_num_ne_zero_of_ne_zero {q : ℚ} {n d : ℤ} (hq : q ≠ 0) (hqnd : q = n /. d) : n ≠ 0 :=
                       /-
                         q : Rat
                         n d : Int
                         hq : Ne q 0
                         hqnd : Eq q (Rat.divInt n d)
                         this : Eq n 0
                         ⊢ Eq q 0
                       -/
  fun this => hq <| by simpa [this] using hqnd
                       /-
                         🎉 no goals
                       -/


theorem mk_denom_ne_zero_of_ne_zero {q : ℚ} {n d : ℤ} (hq : q ≠ 0) (hqnd : q = n /. d) : d ≠ 0 :=
                       /-
                         q : Rat
                         n d : Int
                         hq : Ne q 0
                         hqnd : Eq q (Rat.divInt n d)
                         this : Eq d 0
                         ⊢ Eq q 0
                       -/
  fun this => hq <| by simpa [this] using hqnd
                       /-
                         🎉 no goals
                       -/


theorem divInt_ne_zero_of_ne_zero {n d : ℤ} (h : n ≠ 0) (hd : d ≠ 0) : n /. d ≠ 0 :=
  (divInt_ne_zero hd).mpr h


protected lemma nonneg_antisymm : 0 ≤ q → 0 ≤ -q → q = 0 := by
  /-
    q : Rat
    ⊢ LE.le 0 q → LE.le 0 (Neg.neg q) → Eq q 0
  -/
  simp_rw [← num_eq_zero, Int.le_antisymm_iff, ← num_nonneg, num_neg_eq_neg_num, Int.neg_nonneg]
  /-
    q : Rat
    ⊢ LE.le 0 q.num → LE.le q.num 0 → And (LE.le q.num 0) (LE.le 0 q.num)
  -/
  tauto
  /-
    🎉 no goals
  -/


protected lemma nonneg_total (a : ℚ) : 0 ≤ a ∨ 0 ≤ -a := by
  /-
    a : Rat
    ⊢ Or (LE.le 0 a) (LE.le 0 (Neg.neg a))
  -/
  simp_rw [← num_nonneg, num_neg_eq_neg_num, Int.neg_nonneg]; exact Int.le_total _ _
                                                              /-
                                                                🎉 no goals
                                                              -/


protected theorem add_divInt (a b c : ℤ) : (a + b) /. c = a /. c + b /. c :=
                       /-
                         a b c : Int
                         h : Eq c 0
                         ⊢ Eq (Rat.divInt (HAdd.hAdd a b) c) (HAdd.hAdd (Rat.divInt a c) (Rat.divInt b  …
                       -/
  if h : c = 0 then by simp [h]
                       /-
                         🎉 no goals
                       -/
  else by
    /-
      a b c : Int
      h : Not (Eq c 0)
      ⊢ Eq (Rat.divInt (HAdd.hAdd a b) c) (HAdd.hAdd (Rat.divInt a c) (Rat.divInt b  …
    -/
    rw [divInt_add_divInt _ _ h h, divInt_eq_iff h (Int.mul_ne_zero h h)]
    /-
      a b c : Int
      h : Not (Eq c 0)
      ⊢ Eq (HMul.hMul (HAdd.hAdd a b) (HMul.hMul c c)) (HMul.hMul (HAdd.hAdd (HMul.h …
    -/
    simp [Int.add_mul, Int.mul_assoc]
    /-
      🎉 no goals
    -/


                                                             /-
                                                               n d : Int
                                                               ⊢ Eq (Rat.divInt n d) (HDiv.hDiv ↑n ↑d)
                                                             -/
theorem divInt_eq_div (n d : ℤ) : n /. d = (n : ℚ) / d := by simp [div_def']
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                                     /-
                                                                       n d : Int
                                                                       ⊢ Eq (HDiv.hDiv ↑n ↑d) (Rat.divInt n d)
                                                                     -/
lemma intCast_div_eq_divInt (n d : ℤ) : (n : ℚ) / (d) = n /. d := by rw [divInt_eq_div]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem natCast_div_eq_divInt (n d : ℕ) : (n : ℚ) / d = n /. d := Rat.intCast_div_eq_divInt n d


theorem divInt_mul_divInt_cancel {x : ℤ} (hx : x ≠ 0) (n d : ℤ) : n /. x * (x /. d) = n /. d := by
  /-
    x : Int
    hx : Ne x 0
    n d : Int
    ⊢ Eq (HMul.hMul (Rat.divInt n x) (Rat.divInt x d)) (Rat.divInt n d)
  -/
  by_cases hd : d = 0
    /-
      case pos
      x : Int
      hx : Ne x 0
      n d : Int
      hd : Eq d 0
      ⊢ Eq (HMul.hMul (Rat.divInt n x) (Rat.divInt x d)) (Rat.divInt n d)
    -/
  · rw [hd]
    /-
      case pos
      x : Int
      hx : Ne x 0
      n d : Int
      hd : Eq d 0
      ⊢ Eq (HMul.hMul (Rat.divInt n x) (Rat.divInt x 0)) (Rat.divInt n 0)
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case neg
    x : Int
    hx : Ne x 0
    n d : Int
    hd : Not (Eq d 0)
    ⊢ Eq (HMul.hMul (Rat.divInt n x) (Rat.divInt x d)) (Rat.divInt n d)
  -/
  rw [divInt_mul_divInt _ _ hx hd, x.mul_comm, divInt_mul_right hx]
  /-
    🎉 no goals
  -/


theorem coe_int_num_of_den_eq_one {q : ℚ} (hq : q.den = 1) : (q.num : ℚ) = q := by
  /-
    q : Rat
    hq : Eq q.den 1
    ⊢ Eq (↑q.num) q
  -/
  conv_rhs => rw [← num_divInt_den q, hq]
  /-
    q : Rat
    hq : Eq q.den 1
    ⊢ Eq (↑q.num) (Rat.divInt q.num ↑1)
  -/
  rw [intCast_eq_divInt]
  /-
    q : Rat
    hq : Eq q.den 1
    ⊢ Eq (Rat.divInt q.num 1) (Rat.divInt q.num ↑1)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma eq_num_of_isInt {q : ℚ} (h : q.isInt) : q = q.num := by
  /-
    q : Rat
    h : Eq q.isInt Bool.true
    ⊢ Eq q ↑q.num
  -/
  rw [Rat.isInt, Nat.beq_eq_true_eq] at h
  /-
    q : Rat
    h : Eq q.den 1
    ⊢ Eq q ↑q.num
  -/
  exact (Rat.coe_int_num_of_den_eq_one h).symm
  /-
    🎉 no goals
  -/


theorem den_eq_one_iff (r : ℚ) : r.den = 1 ↔ ↑r.num = r :=
  ⟨Rat.coe_int_num_of_den_eq_one, fun h => h ▸ Rat.den_intCast r.num⟩


instance canLift : CanLift ℚ ℤ (↑) fun q => q.den = 1 :=
  ⟨fun q hq => ⟨q.num, coe_int_num_of_den_eq_one hq⟩⟩


@[deprecated (since := "2024-04-05")] alias coe_int_eq_divInt := intCast_eq_divInt

@[deprecated (since := "2024-04-05")] alias coe_int_div_eq_divInt := intCast_div_eq_divInt

-- Will be subsumed by `Int.coe_inj` after we have defined
-- `LinearOrderedField ℚ` (which implies characteristic zero).

theorem coe_int_inj (m n : ℤ) : (m : ℚ) = n ↔ m = n :=
  ⟨congr_arg num, congr_arg _⟩


