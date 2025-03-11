@[simp]
theorem zero_oreDiv' (s : S) : (0 : R) /ₒ s = 0 := by
  /-
    R : Type u_1
    inst✝¹ : MonoidWithZero R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (OreLocalization.oreDiv 0 s) 0
  -/
  rw [OreLocalization.zero_def, oreDiv_eq_iff]
  /-
    R : Type u_1
    inst✝¹ : MonoidWithZero R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    s : Subtype fun x => Membership.mem S x
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u 0) (HSMul.hSMul v 0)) …
  -/
  exact ⟨s, 1, by simp [Submonoid.smul_def]⟩
  /-
    🎉 no goals
  -/


instance : MonoidWithZero R[S⁻¹] where
  zero_mul x := by
    /-
      R : Type u_1
      inst✝¹ : MonoidWithZero R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      x : OreLocalization S R
      ⊢ Eq (HMul.hMul 0 x) 0
    -/
    induction' x using OreLocalization.ind with r s
    /-
      case c
      R : Type u_1
      inst✝¹ : MonoidWithZero R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      r : R
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul 0 (OreLocalization.oreDiv r s)) 0
    -/
    rw [OreLocalization.zero_def, oreDiv_mul_char 0 r 1 s 0 1 (by simp), zero_mul, one_mul]
    /-
      🎉 no goals
    -/
  mul_zero x := by
    /-
      R : Type u_1
      inst✝¹ : MonoidWithZero R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      x : OreLocalization S R
      ⊢ Eq (HMul.hMul x 0) 0
    -/
    induction' x using OreLocalization.ind with r s
    /-
      case c
      R : Type u_1
      inst✝¹ : MonoidWithZero R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      r : R
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r s) 0) 0
    -/
    rw [OreLocalization.zero_def, mul_div_one, mul_zero, zero_oreDiv', zero_oreDiv']
    /-
      🎉 no goals
    -/


instance : CommMonoidWithZero R[S⁻¹] where
  __ := inferInstanceAs (MonoidWithZero R[S⁻¹])
  __ := inferInstanceAs (CommMonoid R[S⁻¹])


private def add'' (r₁ : X) (s₁ : S) (r₂ : X) (s₂ : S) : X[S⁻¹] :=
  (oreDenom (s₁ : R) s₂ • r₁ + oreNum (s₁ : R) s₂ • r₂) /ₒ (oreDenom (s₁ : R) s₂ * s₁)


private theorem add''_char (r₁ : X) (s₁ : S) (r₂ : X) (s₂ : S) (rb : R) (sb : R)
    (hb : sb * s₁ = rb * s₂) (h : sb * s₁ ∈ S) :
    add'' r₁ s₁ r₂ s₂ = (sb • r₁ + rb • r₂) /ₒ ⟨sb * s₁, h⟩ := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ⊢ Eq (OreLocalization.add'' r₁ s₁ r₂ s₂) (OreLocalization.oreDiv (HAdd.hAdd (H …
  -/
  simp only [add'']
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul (OreLocalization.oreDenom …
  -/
  have ha := ore_eq (s₁ : R) s₂
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ha : Eq (HMul.hMul ↑(OreLocalization.oreDenom (↑s₁) s₂) ↑s₁) (HMul.hMul (OreLo …
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul (OreLocalization.oreDenom …
  -/
  generalize oreNum (s₁ : R) s₂ = ra at *
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    ha : Eq (HMul.hMul ↑(OreLocalization.oreDenom (↑s₁) s₂) ↑s₁) (HMul.hMul ra ↑s₂)
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul (OreLocalization.oreDenom …
  -/
  generalize oreDenom (s₁ : R) s₂ = sa at *
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₁) (HSMul.hSMul ra r₂ …
  -/
  rw [oreDiv_eq_iff]
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u (HAdd.hAdd (HSMul.hSM …
  -/
  rcases oreCondition sb sa with ⟨rc, sc, hc⟩
  have : sc * rb * s₂ = rc * ra * s₂ := by
    rw [mul_assoc rc, ← ha, ← mul_assoc, ← hc, mul_assoc, mul_assoc, hb]
  /-
    case mk.mk
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
    this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u (HAdd.hAdd (HSMul.hSM …
  -/
  rcases ore_right_cancel _ _ s₂ this with ⟨sd, hd⟩
  /-
    case mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
    this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
    sd : Subtype fun x => Membership.mem S x
    hd : Eq (HMul.hMul (↑sd) (HMul.hMul (↑sc) rb)) (HMul.hMul (↑sd) (HMul.hMul rc  …
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u (HAdd.hAdd (HSMul.hSM …
  -/
  use sd * sc
  /-
    case h
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
    this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
    sd : Subtype fun x => Membership.mem S x
    hd : Eq (HMul.hMul (↑sd) (HMul.hMul (↑sc) rb)) (HMul.hMul (↑sd) (HMul.hMul rc  …
    ⊢ Exists fun v => And (Eq (HSMul.hSMul (HMul.hMul sd sc) (HAdd.hAdd (HSMul.hSM …
  -/
  use sd * rc
  /-
    case h
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
    this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
    sd : Subtype fun x => Membership.mem S x
    hd : Eq (HMul.hMul (↑sd) (HMul.hMul (↑sc) rb)) (HMul.hMul (↑sd) (HMul.hMul rc  …
    ⊢ And (Eq (HSMul.hSMul (HMul.hMul sd sc) (HAdd.hAdd (HSMul.hSMul sb r₁) (HSMul …
  -/
  simp only [smul_add, smul_smul, Submonoid.smul_def, Submonoid.coe_mul]
  /-
    case h
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    rb sb : R
    hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
    h : Membership.mem S (HMul.hMul sb ↑s₁)
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
    this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
    sd : Subtype fun x => Membership.mem S x
    hd : Eq (HMul.hMul (↑sd) (HMul.hMul (↑sc) rb)) (HMul.hMul (↑sd) (HMul.hMul rc  …
    ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul (HMul.hMul ↑sd ↑sc) sb) r₁) (HSMu …
  -/
  constructor
    /-
      case h.left
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      rb sb : R
      hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
      h : Membership.mem S (HMul.hMul sb ↑s₁)
      ra : R
      sa : Subtype fun x => Membership.mem S x
      ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
      this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) (HMul.hMul (↑sc) rb)) (HMul.hMul (↑sd) (HMul.hMul rc  …
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul (HMul.hMul (HMul.hMul ↑sd ↑sc) sb) r₁) (HSMul.hSM …
    -/
  · rw [mul_assoc _ _ rb, hd, mul_assoc, hc, mul_assoc, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case h.right
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      rb sb : R
      hb : Eq (HMul.hMul sb ↑s₁) (HMul.hMul rb ↑s₂)
      h : Membership.mem S (HMul.hMul sb ↑s₁)
      ra : R
      sa : Subtype fun x => Membership.mem S x
      ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul (↑sc) sb) (HMul.hMul rc ↑sa)
      this : Eq (HMul.hMul (HMul.hMul (↑sc) rb) ↑s₂) (HMul.hMul (HMul.hMul rc ra) ↑s₂)
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) (HMul.hMul (↑sc) rb)) (HMul.hMul (↑sd) (HMul.hMul rc  …
      ⊢ Eq (HMul.hMul (HMul.hMul ↑sd ↑sc) (HMul.hMul sb ↑s₁)) (HMul.hMul (HMul.hMul  …
    -/
  · rw [mul_assoc, ← mul_assoc (sc : R), hc, mul_assoc, mul_assoc]
    /-
      🎉 no goals
    -/


private def add' (r₂ : X) (s₂ : S) : X[S⁻¹] → X[S⁻¹] :=
  (--plus tilde
      Quotient.lift
      fun r₁s₁ : X × S => add'' r₁s₁.1 r₁s₁.2 r₂ s₂) <| by
    -- Porting note: `assoc_rw` & `noncomm_ring` were not ported yet
    /-
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ ∀ (a b : Prod X (Subtype fun x => Membership.mem S x)), HasEquiv.Equiv a b → …
    -/
    rintro ⟨r₁', s₁'⟩ ⟨r₁, s₁⟩ ⟨sb, rb, hb, hb'⟩
    -- s*, r*
    /-
      case mk.mk.intro.intro.intro
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb { fst := r₁, snd := s₁ }.1) (HSMul.hSMul rb { fst := r …
      hb' : Eq (HMul.hMul ↑sb ↑{ fst := r₁, snd := s₁ }.2) (HMul.hMul rb ↑{ fst := r …
      ⊢ Eq (OreLocalization.add'' { fst := r₁', snd := s₁' }.1 { fst := r₁', snd :=  …
    -/
    rcases oreCondition (s₁' : R) s₂ with ⟨rc, sc, hc⟩
    --s~~, r~~
    /-
      case mk.mk.intro.intro.intro.mk.mk
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb { fst := r₁, snd := s₁ }.1) (HSMul.hSMul rb { fst := r …
      hb' : Eq (HMul.hMul ↑sb ↑{ fst := r₁, snd := s₁ }.2) (HMul.hMul rb ↑{ fst := r …
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      ⊢ Eq (OreLocalization.add'' { fst := r₁', snd := s₁' }.1 { fst := r₁', snd :=  …
    -/
    rcases oreCondition rb sc with ⟨rd, sd, hd⟩
    -- s#, r#
    /-
      case mk.mk.intro.intro.intro.mk.mk.mk.mk
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb { fst := r₁, snd := s₁ }.1) (HSMul.hSMul rb { fst := r …
      hb' : Eq (HMul.hMul ↑sb ↑{ fst := r₁, snd := s₁ }.2) (HMul.hMul rb ↑{ fst := r …
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      ⊢ Eq (OreLocalization.add'' { fst := r₁', snd := s₁' }.1 { fst := r₁', snd :=  …
    -/
    dsimp at *
    /-
      case mk.mk.intro.intro.intro.mk.mk.mk.mk
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      ⊢ Eq (OreLocalization.add'' r₁' s₁' r₂ s₂) (OreLocalization.add'' r₁ s₁ r₂ s₂)
    -/
    rw [add''_char _ _ _ _ rc sc hc (sc * s₁').2]
    have : sd * sb * s₁ = rd * rc * s₂ := by
      rw [mul_assoc, hb', ← mul_assoc, hd, mul_assoc, hc, ← mul_assoc]
    /-
      case mk.mk.intro.intro.intro.mk.mk.mk.mk
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      this : Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul (HMul.hMul rd rc) ↑s₂)
      ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul (↑sc) r₁') (HSMul.hSMul r …
    -/
    rw [add''_char _ _ _ _ (rd * rc : R) (sd * sb) this (sd * sb * s₁).2]
    /-
      case mk.mk.intro.intro.intro.mk.mk.mk.mk
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      this : Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul (HMul.hMul rd rc) ↑s₂)
      ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul (↑sc) r₁') (HSMul.hSMul r …
    -/
    rw [mul_smul, ← Submonoid.smul_def sb, hb, smul_smul, hd, oreDiv_eq_iff]
    /-
      case mk.mk.intro.intro.intro.mk.mk.mk.mk
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      this : Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul (HMul.hMul rd rc) ↑s₂)
      ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u (HAdd.hAdd (HSMul.hSM …
    -/
    use 1
    /-
      case h
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      this : Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul (HMul.hMul rd rc) ↑s₂)
      ⊢ Exists fun v => And (Eq (HSMul.hSMul 1 (HAdd.hAdd (HSMul.hSMul (HMul.hMul rd …
    -/
    use rd
    /-
      case h
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      this : Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul (HMul.hMul rd rc) ↑s₂)
      ⊢ And (Eq (HSMul.hSMul 1 (HAdd.hAdd (HSMul.hSMul (HMul.hMul rd ↑sc) r₁') (HSMu …
    -/
    simp only [mul_smul, smul_add, one_smul, OneMemClass.coe_one, one_mul, true_and]
    /-
      case h
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : X
      s₁' : Subtype fun x => Membership.mem S x
      r₁ : X
      s₁ sb : Subtype fun x => Membership.mem S x
      rb : R
      hb : Eq (HSMul.hSMul sb r₁) (HSMul.hSMul rb r₁')
      hb' : Eq (HMul.hMul ↑sb ↑s₁) (HMul.hMul rb ↑s₁')
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑s₁') (HMul.hMul rc ↑s₂)
      rd : R
      sd : Subtype fun x => Membership.mem S x
      hd : Eq (HMul.hMul (↑sd) rb) (HMul.hMul rd ↑sc)
      this : Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul (HMul.hMul rd rc) ↑s₂)
      ⊢ Eq (HMul.hMul (HMul.hMul ↑sd ↑sb) ↑s₁) (HMul.hMul rd (HMul.hMul ↑sc ↑s₁'))
    -/
    rw [this, hc, mul_assoc]
    /-
      🎉 no goals
    -/


/-- The addition on the Ore localization. -/
@[irreducible]
private def add : X[S⁻¹] → X[S⁻¹] → X[S⁻¹] := fun x =>
  Quotient.lift (fun rs : X × S => add' rs.1 rs.2 x)
    (by
      /-
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        x : OreLocalization S X
        ⊢ ∀ (a b : Prod X (Subtype fun x => Membership.mem S x)), HasEquiv.Equiv a b → …
      -/
      rintro ⟨r₁, s₁⟩ ⟨r₂, s₂⟩ ⟨sb, rb, hb, hb'⟩
      /-
        case mk.mk.intro.intro.intro
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        x : OreLocalization S X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb { fst := r₂, snd := s₂ }.1) (HSMul.hSMul rb { fst := r …
        hb' : Eq (HMul.hMul ↑sb ↑{ fst := r₂, snd := s₂ }.2) (HMul.hMul rb ↑{ fst := r …
        ⊢ Eq ((fun rs => OreLocalization.add' rs.1 rs.2 x) { fst := r₁, snd := s₁ }) ( …
      -/
      induction' x with r₃ s₃
      /-
        case mk.mk.intro.intro.intro.c
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb { fst := r₂, snd := s₂ }.1) (HSMul.hSMul rb { fst := r …
        hb' : Eq (HMul.hMul ↑sb ↑{ fst := r₂, snd := s₂ }.2) (HMul.hMul rb ↑{ fst := r …
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq ((fun rs => OreLocalization.add' rs.1 rs.2 (OreLocalization.oreDiv r₃ s₃) …
      -/
      show add'' _ _ _ _ = add'' _ _ _ _
      /-
        case mk.mk.intro.intro.intro.c
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb { fst := r₂, snd := s₂ }.1) (HSMul.hSMul rb { fst := r …
        hb' : Eq (HMul.hMul ↑sb ↑{ fst := r₂, snd := s₂ }.2) (HMul.hMul rb ↑{ fst := r …
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq (OreLocalization.add'' { fst := r₃, snd := s₃ }.1 { fst := r₃, snd := s₃  …
      -/
      dsimp only at *
      /-
        case mk.mk.intro.intro.intro.c
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        ⊢ Eq (OreLocalization.add'' r₃ s₃ r₁ s₁) (OreLocalization.add'' r₃ s₃ r₂ s₂)
      -/
      rcases oreCondition (s₃ : R) s₂ with ⟨rc, sc, hc⟩
      /-
        case mk.mk.intro.intro.intro.c.mk.mk
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        ⊢ Eq (OreLocalization.add'' r₃ s₃ r₁ s₁) (OreLocalization.add'' r₃ s₃ r₂ s₂)
      -/
      rcases oreCondition rc sb with ⟨rd, sd, hd⟩
      have : rd * rb * s₁ = sd * sc * s₃ := by
        rw [mul_assoc, ← hb', ← mul_assoc, ← hd, mul_assoc, ← hc, mul_assoc]
      /-
        case mk.mk.intro.intro.intro.c.mk.mk.mk.mk
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ Eq (OreLocalization.add'' r₃ s₃ r₁ s₁) (OreLocalization.add'' r₃ s₃ r₂ s₂)
      -/
      rw [add''_char _ _ _ _ rc sc hc (sc * s₃).2]
      /-
        case mk.mk.intro.intro.intro.c.mk.mk.mk.mk
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ Eq (OreLocalization.add'' r₃ s₃ r₁ s₁) (OreLocalization.oreDiv (HAdd.hAdd (H …
      -/
      rw [add''_char _ _ _ _ _ _ this.symm (sd * sc * s₃).2]
      /-
        case mk.mk.intro.intro.intro.c.mk.mk.mk.mk
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul (HMul.hMul ↑sd ↑sc) r₃) ( …
      -/
      refine oreDiv_eq_iff.mpr ?_
      /-
        case mk.mk.intro.intro.intro.c.mk.mk.mk.mk
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u (HAdd.hAdd (HSMul.hSM …
      -/
      simp only [Submonoid.mk_smul, smul_add]
      /-
        case mk.mk.intro.intro.intro.c.mk.mk.mk.mk
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ Exists fun u => Exists fun v => And (Eq (HAdd.hAdd (HSMul.hSMul u (HSMul.hSM …
      -/
      use sd, 1
      /-
        case h
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ And (Eq (HAdd.hAdd (HSMul.hSMul sd (HSMul.hSMul (↑sc) r₃)) (HSMul.hSMul sd ( …
      -/
      simp only [one_smul, one_mul, mul_smul, ← hb, Submonoid.smul_def, ← mul_assoc, and_true]
      /-
        case h
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type u_2
        inst✝¹ : AddMonoid X
        inst✝ : DistribMulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ sb : Subtype fun x => Membership.mem S x
        rb : R
        hb : Eq (HSMul.hSMul sb r₂) (HSMul.hSMul rb r₁)
        hb' : Eq (HMul.hMul ↑sb ↑s₂) (HMul.hMul rb ↑s₁)
        r₃ : X
        s₃ : Subtype fun x => Membership.mem S x
        rc : R
        sc : Subtype fun x => Membership.mem S x
        hc : Eq (HMul.hMul ↑sc ↑s₃) (HMul.hMul rc ↑s₂)
        rd : R
        sd : Subtype fun x => Membership.mem S x
        hd : Eq (HMul.hMul (↑sd) rc) (HMul.hMul rd ↑sb)
        this : Eq (HMul.hMul (HMul.hMul rd rb) ↑s₁) (HMul.hMul (HMul.hMul ↑sd ↑sc) ↑s₃)
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul (↑sd) (HSMul.hSMul (↑sc) r₃)) (HSMul.hSMul (↑sd)  …
      -/
      simp only [smul_smul, hd])
      /-
        🎉 no goals
      -/


instance : Add X[S⁻¹] :=
  ⟨add⟩


theorem oreDiv_add_oreDiv {r r' : X} {s s' : S} :
    r /ₒ s + r' /ₒ s' =
      (oreDenom (s : R) s' • r + oreNum (s : R) s' • r') /ₒ (oreDenom (s : R) s' * s) := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r r' : X
    s s' : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s')) ( …
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


theorem oreDiv_add_char' {r r' : X} (s s' : S) (rb : R) (sb : R)
    (h : sb * s = rb * s') (h' : sb * s ∈ S) :
    r /ₒ s + r' /ₒ s' = (sb • r + rb • r') /ₒ ⟨sb * s, h'⟩ := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r r' : X
    s s' : Subtype fun x => Membership.mem S x
    rb sb : R
    h : Eq (HMul.hMul sb ↑s) (HMul.hMul rb ↑s')
    h' : Membership.mem S (HMul.hMul sb ↑s)
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s')) ( …
  -/
  with_unfolding_all exact add''_char r s r' s' rb sb h h'
  /-
    🎉 no goals
  -/


/-- A characterization of the addition on the Ore localizaion, allowing for arbitrary Ore
numerator and Ore denominator. -/
theorem oreDiv_add_char {r r' : X} (s s' : S) (rb : R) (sb : S) (h : sb * s = rb * s') :
    r /ₒ s + r' /ₒ s' = (sb • r + rb • r') /ₒ (sb * s) :=
  oreDiv_add_char' s s' rb sb h (sb * s).2


/-- Another characterization of the addition on the Ore localization, bundling up all witnesses
and conditions into a sigma type. -/
def oreDivAddChar' (r r' : X) (s s' : S) :
    Σ'r'' : R,
      Σ's'' : S, s'' * s = r'' * s' ∧ r /ₒ s + r' /ₒ s' = (s'' • r + r'' • r') /ₒ (s'' * s) :=
  ⟨oreNum (s : R) s', oreDenom (s : R) s', ore_eq (s : R) s', oreDiv_add_oreDiv⟩


@[simp]
theorem add_oreDiv {r r' : X} {s : S} : r /ₒ s + r' /ₒ s = (r + r') /ₒ s := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r r' : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s)) (O …
  -/
  simp [oreDiv_add_char s s 1 1 (by simp)]
  /-
    🎉 no goals
  -/


protected theorem add_assoc (x y z : X[S⁻¹]) : x + y + z = x + (y + z) := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    x y z : OreLocalization S X
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd x y) z) (HAdd.hAdd x (HAdd.hAdd y z))
  -/
  induction' x with r₁ s₁
  /-
    case c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    y z : OreLocalization S X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) y) z) (HAdd.hAdd (Or …
  -/
  induction' y with r₂ s₂
  /-
    case c.c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    z : OreLocalization S X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) (OreLocalization.ore …
  -/
  induction' z with r₃ s₃
  /-
    case c.c.c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) (OreLocalization.ore …
  -/
  rcases oreDivAddChar' r₁ r₂ s₁ s₂ with ⟨ra, sa, ha, ha'⟩; rw [ha']; clear ha'
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₁) (HSMul. …
  -/
  rcases oreDivAddChar' (sa • r₁ + ra • r₂) r₃ (sa * s₁) s₃ with ⟨rc, sc, hc, q⟩; rw [q]; clear q
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul ↑sc ↑(HMul.hMul sa s₁)) (HMul.hMul rc ↑s₃)
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sc (HAdd.hAdd (HSMul.hSMu …
  -/
  simp only [smul_add, mul_assoc, add_assoc]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul ↑sc ↑(HMul.hMul sa s₁)) (HMul.hMul rc ↑s₃)
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sc (HSMul.hSMul sa r₁)) ( …
  -/
  simp_rw [← add_oreDiv, ← OreLocalization.expand']
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul ↑sc ↑(HMul.hMul sa s₁)) (HMul.hMul rc ↑s₃)
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) (HAdd.hAdd (OreLocalization.ore …
  -/
  congr 2
    /-
      case c.c.c.mk.mk.intro.mk.mk.intro.e_a.e_a
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₃ : X
      s₃ : Subtype fun x => Membership.mem S x
      ra : R
      sa : Subtype fun x => Membership.mem S x
      ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑(HMul.hMul sa s₁)) (HMul.hMul rc ↑s₃)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul ra r₂) (HMul.hMul sa s₁)) (OreLocali …
    -/
  · rw [OreLocalization.expand r₂ s₂ ra (ha.symm ▸ (sa * s₁).2)]; congr; ext; exact ha
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    /-
      case c.c.c.mk.mk.intro.mk.mk.intro.e_a.e_a
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddMonoid X
      inst✝ : DistribMulAction R X
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      r₃ : X
      s₃ : Subtype fun x => Membership.mem S x
      ra : R
      sa : Subtype fun x => Membership.mem S x
      ha : Eq (HMul.hMul ↑sa ↑s₁) (HMul.hMul ra ↑s₂)
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul ↑sc ↑(HMul.hMul sa s₁)) (HMul.hMul rc ↑s₃)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul rc r₃) (HMul.hMul sc (HMul.hMul sa s …
    -/
  · rw [OreLocalization.expand r₃ s₃ rc (hc.symm ▸ (sc * (sa * s₁)).2)]; congr; ext; exact hc
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


@[simp]
theorem zero_oreDiv (s : S) : (0 : X) /ₒ s = 0 := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (OreLocalization.oreDiv 0 s) 0
  -/
  rw [OreLocalization.zero_def, oreDiv_eq_iff]
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    s : Subtype fun x => Membership.mem S x
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u 0) (HSMul.hSMul v 0)) …
  -/
  exact ⟨s, 1, by simp⟩
  /-
    🎉 no goals
  -/


protected theorem zero_add (x : X[S⁻¹]) : 0 + x = x := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    x : OreLocalization S X
    ⊢ Eq (HAdd.hAdd 0 x) x
  -/
  induction x
  /-
    case c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r✝ : X
    s✝ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd 0 (OreLocalization.oreDiv r✝ s✝)) (OreLocalization.oreDiv r✝ s✝)
  -/
  rw [← zero_oreDiv, add_oreDiv]; simp
                                  /-
                                    🎉 no goals
                                  -/


protected theorem add_zero (x : X[S⁻¹]) : x + 0 = x := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    x : OreLocalization S X
    ⊢ Eq (HAdd.hAdd x 0) x
  -/
  induction x
  /-
    case c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r✝ : X
    s✝ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r✝ s✝) 0) (OreLocalization.oreDiv r✝ s✝)
  -/
  rw [← zero_oreDiv, add_oreDiv]; simp
                                  /-
                                    🎉 no goals
                                  -/


@[irreducible]
private def nsmul : ℕ → X[S⁻¹] → X[S⁻¹] := nsmulRec


instance : AddMonoid X[S⁻¹] where
    add_assoc := OreLocalization.add_assoc
    zero_add := OreLocalization.zero_add
    add_zero := OreLocalization.add_zero
    nsmul := nsmul
                       /-
                         R : Type u_1
                         inst✝³ : Monoid R
                         S : Submonoid R
                         inst✝² : OreLocalization.OreSet S
                         X : Type u_2
                         inst✝¹ : AddMonoid X
                         inst✝ : DistribMulAction R X
                         x✝ : OreLocalization S X
                         ⊢ Eq (OreLocalization.nsmul 0 x✝) 0
                       -/
    nsmul_zero _ := by with_unfolding_all rfl
                       /-
                         🎉 no goals
                       -/
                         /-
                           R : Type u_1
                           inst✝³ : Monoid R
                           S : Submonoid R
                           inst✝² : OreLocalization.OreSet S
                           X : Type u_2
                           inst✝¹ : AddMonoid X
                           inst✝ : DistribMulAction R X
                           x✝¹ : Nat
                           x✝ : OreLocalization S X
                           ⊢ Eq (OreLocalization.nsmul (HAdd.hAdd x✝¹ 1) x✝) (HAdd.hAdd (OreLocalization. …
                         -/
    nsmul_succ _ _ := by with_unfolding_all rfl
                         /-
                           🎉 no goals
                         -/


protected theorem smul_zero (x : R[S⁻¹]) : x • (0 : X[S⁻¹]) = 0 := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    x : OreLocalization S R
    ⊢ Eq (HSMul.hSMul x 0) 0
  -/
  induction' x with r s
  /-
    case c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r s) 0) 0
  -/
  rw [OreLocalization.zero_def, smul_div_one, smul_zero, zero_oreDiv, zero_oreDiv]
  /-
    🎉 no goals
  -/


protected theorem smul_add (z : R[S⁻¹]) (x y : X[S⁻¹]) :
    z • (x + y) = z • x + z • y := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    z : OreLocalization S R
    x y : OreLocalization S X
    ⊢ Eq (HSMul.hSMul z (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul z x) (HSMul.hSMul …
  -/
  induction' x with r₁ s₁
  /-
    case c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    z : OreLocalization S R
    y : OreLocalization S X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul z (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) y)) (HAdd.hAdd ( …
  -/
  induction' y with r₂ s₂
  /-
    case c.c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    z : OreLocalization S R
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul z (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) (OreLocalization …
  -/
  induction' z with r₃ s₃
  /-
    case c.c.c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₃ s₃) (HAdd.hAdd (OreLocalization.o …
  -/
  rcases oreDivAddChar' r₁ r₂ s₁ s₂ with ⟨ra, sa, ha, ha'⟩; rw [ha']; clear ha'; norm_cast at ha
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (↑(HMul.hMul sa s₁)) (HMul.hMul ra ↑s₂)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₃ s₃) (OreLocalization.oreDiv (HAdd …
  -/
  rw [OreLocalization.expand' r₁ s₁ sa]
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (↑(HMul.hMul sa s₁)) (HMul.hMul ra ↑s₂)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₃ s₃) (OreLocalization.oreDiv (HAdd …
  -/
  rw [OreLocalization.expand r₂ s₂ ra (by rw [← ha]; apply SetLike.coe_mem)]
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (↑(HMul.hMul sa s₁)) (HMul.hMul ra ↑s₂)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₃ s₃) (OreLocalization.oreDiv (HAdd …
  -/
  rw [← Subtype.coe_eq_of_eq_mk ha]
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (↑(HMul.hMul sa s₁)) (HMul.hMul ra ↑s₂)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₃ s₃) (OreLocalization.oreDiv (HAdd …
  -/
  repeat rw [oreDiv_smul_oreDiv]
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddMonoid X
    inst✝ : DistribMulAction R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : X
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (↑(HMul.hMul sa s₁)) (HMul.hMul ra ↑s₂)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (OreLocalization.oreNum r₃ (HMul.hMu …
  -/
  simp only [smul_add, add_oreDiv]
  /-
    🎉 no goals
  -/


instance : DistribMulAction R[S⁻¹] X[S⁻¹] where
  smul_zero := OreLocalization.smul_zero
  smul_add := OreLocalization.smul_add


instance {R₀} [Monoid R₀] [MulAction R₀ X] [MulAction R₀ R]
    [IsScalarTower R₀ R X] [IsScalarTower R₀ R R] :
    DistribMulAction R₀ X[S⁻¹] where
                    /-
                      R : Type u_1
                      inst✝⁸ : Monoid R
                      S : Submonoid R
                      inst✝⁷ : OreLocalization.OreSet S
                      X : Type u_2
                      inst✝⁶ : AddMonoid X
                      inst✝⁵ : DistribMulAction R X
                      R₀ : Type ?u.42101
                      inst✝⁴ : Monoid R₀
                      inst✝³ : MulAction R₀ X
                      inst✝² : MulAction R₀ R
                      inst✝¹ : IsScalarTower R₀ R X
                      inst✝ : IsScalarTower R₀ R R
                      x✝ : R₀
                      ⊢ Eq (HSMul.hSMul x✝ 0) 0
                    -/
  smul_zero _ := by rw [← smul_one_oreDiv_one_smul, smul_zero]
                    /-
                      🎉 no goals
                    -/
                       /-
                         R : Type u_1
                         inst✝⁸ : Monoid R
                         S : Submonoid R
                         inst✝⁷ : OreLocalization.OreSet S
                         X : Type u_2
                         inst✝⁶ : AddMonoid X
                         inst✝⁵ : DistribMulAction R X
                         R₀ : Type ?u.42101
                         inst✝⁴ : Monoid R₀
                         inst✝³ : MulAction R₀ X
                         inst✝² : MulAction R₀ R
                         inst✝¹ : IsScalarTower R₀ R X
                         inst✝ : IsScalarTower R₀ R R
                         x✝² : R₀
                         x✝¹ x✝ : OreLocalization S X
                         ⊢ Eq (HSMul.hSMul x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (HSMul.hSMul x✝² x✝¹) (HS …
                       -/
  smul_add _ _ _ := by simp only [← smul_one_oreDiv_one_smul, smul_add]
                       /-
                         🎉 no goals
                       -/


protected theorem add_comm (x y : X[S⁻¹]) : x + y = y + x := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : DistribMulAction R X
    x y : OreLocalization S X
    ⊢ Eq (HAdd.hAdd x y) (HAdd.hAdd y x)
  -/
  induction' x with r s
  /-
    case c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : DistribMulAction R X
    y : OreLocalization S X
    r : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r s) y) (HAdd.hAdd y (OreLocalization. …
  -/
  induction' y with r' s'
  /-
    case c.c
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : DistribMulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    r' : X
    s' : Subtype fun x => Membership.mem S x
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s')) ( …
  -/
  rcases oreDivAddChar' r r' s s' with ⟨ra, sa, ha, ha'⟩
  /-
    case c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : DistribMulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    r' : X
    s' : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s) (HMul.hMul ra ↑s')
    ha' : Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s' …
    ⊢ Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s')) ( …
  -/
  rw [ha', oreDiv_add_char' s' s _ _ ha.symm (ha ▸ (sa * s).2), add_comm]
  /-
    case c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : DistribMulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    r' : X
    s' : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s) (HMul.hMul ra ↑s')
    ha' : Eq (HAdd.hAdd (OreLocalization.oreDiv r s) (OreLocalization.oreDiv r' s' …
    ⊢ Eq (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul ra r') (HSMul.hSMul sa r) …
  -/
  congr; ext; exact ha
              /-
                🎉 no goals
              -/


instance instAddCommMonoidOreLocalization : AddCommMonoid X[S⁻¹] where
  add_comm := OreLocalization.add_comm


/-- Negation on the Ore localization is defined via negation on the numerator. -/
@[irreducible]
protected def neg : X[S⁻¹] → X[S⁻¹] :=
  liftExpand (fun (r : X) (s : S) => -r /ₒ s) fun r t s ht => by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    /-
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddGroup X
      inst✝ : DistribMulAction R X
      r : X
      t : R
      s : Subtype fun x => Membership.mem S x
      ht : Membership.mem S (HMul.hMul t ↑s)
      ⊢ Eq ((fun r s => OreLocalization.oreDiv (Neg.neg r) s) r s) ((fun r s => OreL …
    -/
    beta_reduce
    /-
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type u_2
      inst✝¹ : AddGroup X
      inst✝ : DistribMulAction R X
      r : X
      t : R
      s : Subtype fun x => Membership.mem S x
      ht : Membership.mem S (HMul.hMul t ↑s)
      ⊢ Eq (OreLocalization.oreDiv (Neg.neg r) s) (OreLocalization.oreDiv (Neg.neg ( …
    -/
    rw [← smul_neg, ← OreLocalization.expand]
    /-
      🎉 no goals
    -/


instance instNegOreLocalization : Neg X[S⁻¹] :=
  ⟨OreLocalization.neg⟩


@[simp]
protected theorem neg_def (r : X) (s : S) : -(r /ₒ s) = -r /ₒ s := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddGroup X
    inst✝ : DistribMulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (Neg.neg (OreLocalization.oreDiv r s)) (OreLocalization.oreDiv (Neg.neg r …
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


protected theorem neg_add_cancel (x : X[S⁻¹]) : -x + x = 0 := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddGroup X
    inst✝ : DistribMulAction R X
    x : OreLocalization S X
    ⊢ Eq (HAdd.hAdd (Neg.neg x) x) 0
  -/
  induction' x with r s; simp
                         /-
                           🎉 no goals
                         -/


/-- `zsmul` of `OreLocalization` -/
@[irreducible]
protected def zsmul : ℤ → X[S⁻¹] → X[S⁻¹] := zsmulRec


unseal OreLocalization.zsmul in
instance instAddGroupOreLocalization : AddGroup X[S⁻¹] where
  neg_add_cancel := OreLocalization.neg_add_cancel
  zsmul := OreLocalization.zsmul


instance : AddCommGroup X[S⁻¹] where
  __ := inferInstanceAs (AddGroup X[S⁻¹])
  __ := inferInstanceAs (AddCommMonoid X[S⁻¹])


