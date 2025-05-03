protected theorem zero_smul (x : X[S⁻¹]) : (0 : R[S⁻¹]) • x = 0 := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul 0 x) 0
  -/
  induction' x with r s
  /-
    case c
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul 0 (OreLocalization.oreDiv r s)) 0
  -/
  rw [OreLocalization.zero_def, oreDiv_smul_char 0 r 1 s 0 1 (by simp)]; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


protected theorem add_smul (y z : R[S⁻¹]) (x : X[S⁻¹]) :
    (y + z) • x = y • x + z • x := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    y z : OreLocalization S R
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd y z) x) (HAdd.hAdd (HSMul.hSMul y x) (HSMul.hSMul …
  -/
  induction' x with r₁ s₁
  /-
    case c
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    y z : OreLocalization S R
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd y z) (OreLocalization.oreDiv r₁ s₁)) (HAdd.hAdd ( …
  -/
  induction' y with r₂ s₂
  /-
    case c.c
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    z : OreLocalization S R
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (OreLocalization.oreDiv r₂ s₂) z) (OreLocalizatio …
  -/
  induction' z with r₃ s₃
  /-
    case c.c.c
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (OreLocalization.oreDiv r₂ s₂) (OreLocalization.o …
  -/
  rcases oreDivAddChar' r₂ r₃ s₂ s₃ with ⟨ra, sa, ha, q⟩
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    q : Eq (HAdd.hAdd (OreLocalization.oreDiv r₂ s₂) (OreLocalization.oreDiv r₃ s₃ …
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (OreLocalization.oreDiv r₂ s₂) (OreLocalization.o …
  -/
  rw [q]
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    q : Eq (HAdd.hAdd (OreLocalization.oreDiv r₂ s₂) (OreLocalization.oreDiv r₃ s₃ …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  clear q
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [OreLocalization.expand' r₂ s₂ sa]
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rcases oreDivSMulChar' (sa • r₂) r₁ (sa * s₂) s₁ with ⟨rb, sb, hb, q⟩
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    q : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul sa r₂) (HMul.hMul sa  …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [q]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    q : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul sa r₂) (HMul.hMul sa  …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  clear q
  have hs₃rasb : sb * ra * s₃ ∈ S := by
    rw [mul_assoc, ← ha]
    norm_cast
    apply SetLike.coe_mem
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [OreLocalization.expand _ _ _ hs₃rasb]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  have ha' : ↑((sb * sa) * s₂) = sb * ra * s₃ := by simp [ha, mul_assoc]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [← Subtype.coe_eq_of_eq_mk ha']
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rcases oreDivSMulChar' ((sb * ra) • r₃) r₁ (sb * sa * s₂) s₁ with ⟨rc, sc, hc, hc'⟩
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HSMul.hSMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc  …
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [hc']
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HSMul.hSMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc  …
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [oreDiv_add_char _ _ 1 sc (by simp [mul_assoc])]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HSMul.hSMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc  …
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HSMul.hSMul sa r₂) (HSMu …
  -/
  rw [OreLocalization.expand' (sa • r₂ + ra • r₃) (sa * s₂) (sc * sb)]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HSMul.hSMul sa r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HSMul.hSMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc  …
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul sc sb) (HAdd …
  -/
  simp only [smul_eq_mul, one_smul, Submonoid.smul_def, mul_add, Submonoid.coe_mul] at hb hc ⊢
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HMul.hMul (↑sa) r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HMul.hMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc ↑s₁)
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HAdd.hAdd (HMul.hMul (HMul.hMul ↑sc …
  -/
  rw [mul_assoc, hb, mul_assoc, ← mul_assoc _ ra, hc, ← mul_assoc, ← add_mul]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HMul.hMul (↑sa) r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HMul.hMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc ↑s₁)
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HMul.hMul (HAdd.hAdd (HMul.hMul (↑s …
  -/
  rw [OreLocalization.smul_cancel']
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    r₁ : X
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul ↑sa ↑s₂) (HMul.hMul ra ↑s₃)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) (HMul.hMul (↑sa) r₂)) (HMul.hMul rb ↑s₁)
    hs₃rasb : Membership.mem S (HMul.hMul (HMul.hMul (↑sb) ra) ↑s₃)
    ha' : Eq (↑(HMul.hMul (HMul.hMul sb sa) s₂)) (HMul.hMul (HMul.hMul (↑sb) ra) ↑ …
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) (HMul.hMul (HMul.hMul (↑sb) ra) r₃)) (HMul.hMul rc ↑s₁)
    hc' : Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑sb) ra …
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (HAdd.hAdd (HMul.hMul (↑sc) rb) rc)  …
  -/
  simp only [add_smul, ← mul_assoc, smul_smul]
  /-
    🎉 no goals
  -/


protected theorem zero_mul (x : R[S⁻¹]) : 0 * x = 0 :=
  OreLocalization.zero_smul x


protected theorem mul_zero (x : R[S⁻¹]) : x * 0 = 0 :=
  OreLocalization.smul_zero x


protected theorem left_distrib (x y z : R[S⁻¹]) : x * (y + z) = x * y + x * z :=
  OreLocalization.smul_add _ _ _


theorem right_distrib (x y z : R[S⁻¹]) : (x + y) * z = x * z + y * z :=
  OreLocalization.add_smul _ _ _


instance : Semiring R[S⁻¹] where
  __ := inferInstanceAs (MonoidWithZero (R[S⁻¹]))
  __ := inferInstanceAs (AddCommMonoid (R[S⁻¹]))
  left_distrib := OreLocalization.left_distrib
  right_distrib := right_distrib


instance : Module R[S⁻¹] X[S⁻¹] where
  add_smul := OreLocalization.add_smul
  zero_smul := OreLocalization.zero_smul


instance {R₀} [Semiring R₀] [Module R₀ X] [Module R₀ R]
    [IsScalarTower R₀ R X] [IsScalarTower R₀ R R] :
    Module R₀ X[S⁻¹] where
                       /-
                         R : Type u_1
                         inst✝⁸ : Semiring R
                         S : Submonoid R
                         inst✝⁷ : OreLocalization.OreSet S
                         X : Type u_2
                         inst✝⁶ : AddCommMonoid X
                         inst✝⁵ : Module R X
                         R₀ : Type ?u.20256
                         inst✝⁴ : Semiring R₀
                         inst✝³ : Module R₀ X
                         inst✝² : Module R₀ R
                         inst✝¹ : IsScalarTower R₀ R X
                         inst✝ : IsScalarTower R₀ R R
                         r s : R₀
                         x : OreLocalization S X
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
  add_smul r s x := by simp only [← smul_one_oreDiv_one_smul, add_smul, ← add_oreDiv]
                       /-
                         🎉 no goals
                       -/
                    /-
                      R : Type u_1
                      inst✝⁸ : Semiring R
                      S : Submonoid R
                      inst✝⁷ : OreLocalization.OreSet S
                      X : Type u_2
                      inst✝⁶ : AddCommMonoid X
                      inst✝⁵ : Module R X
                      R₀ : Type ?u.20256
                      inst✝⁴ : Semiring R₀
                      inst✝³ : Module R₀ X
                      inst✝² : Module R₀ R
                      inst✝¹ : IsScalarTower R₀ R X
                      inst✝ : IsScalarTower R₀ R R
                      x : OreLocalization S X
                      ⊢ Eq (HSMul.hSMul 0 x) 0
                    -/
  zero_smul x := by rw [← smul_one_oreDiv_one_smul, zero_smul, zero_oreDiv, zero_smul]
                    /-
                      🎉 no goals
                    -/


@[simp]
lemma nsmul_eq_nsmul (n : ℕ) (x : X[S⁻¹]) :
    letI inst := OreLocalization.instModuleOfIsScalarTower (R₀ := ℕ) (R := R) (X := X) (S := S)
    HSMul.hSMul (self := @instHSMul _ _ inst.toSMul) n x = n • x := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    n : Nat
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul n x) (HSMul.hSMul n x)
  -/
  letI inst := OreLocalization.instModuleOfIsScalarTower (R₀ := ℕ) (R := R) (X := X) (S := S)
  /-
    R : Type u_1
    inst✝³ : Semiring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommMonoid X
    inst✝ : Module R X
    n : Nat
    x : OreLocalization S X
    inst : Module Nat (OreLocalization S X) := OreLocalization.instModuleOfIsScala …
    ⊢ Eq (HSMul.hSMul n x) (HSMul.hSMul n x)
  -/
  exact congr($(AddCommMonoid.uniqueNatModule.2 inst).smul n x)
  /-
    🎉 no goals
  -/


/-- The ring homomorphism from `R` to `R[S⁻¹]`, mapping `r : R` to the fraction `r /ₒ 1`. -/
@[simps!]
def numeratorRingHom : R →+* R[S⁻¹] where
  __ := numeratorHom
                  /-
                    R : Type u_1
                    inst✝³ : Semiring R
                    S : Submonoid R
                    inst✝² : OreLocalization.OreSet S
                    X : Type u_2
                    inst✝¹ : AddCommMonoid X
                    inst✝ : Module R X
                    ⊢ Eq ((↑__spread✝⁻⁰).toFun 0) 0
                  -/
  map_zero' := by with_unfolding_all exact OreLocalization.zero_def
                  /-
                    🎉 no goals
                  -/
  map_add' _ _ := add_oreDiv.symm


instance {R₀} [CommSemiring R₀] [Algebra R₀ R] : Algebra R₀ R[S⁻¹] where
  __ := inferInstanceAs (Module R₀ R[S⁻¹])
  __ := numeratorRingHom.comp (algebraMap R₀ R)
  commutes' r x := by
    /-
      R : Type u_1
      inst✝⁵ : Semiring R
      S : Submonoid R
      inst✝⁴ : OreLocalization.OreSet S
      X : Type u_2
      inst✝³ : AddCommMonoid X
      inst✝² : Module R X
      R₀ : Type ?u.30734
      inst✝¹ : CommSemiring R₀
      inst✝ : Algebra R₀ R
      r : R₀
      x : OreLocalization S R
      ⊢ Eq (HMul.hMul (__spread✝¹⁻⁰ r) x) (HMul.hMul x (__spread✝¹⁻⁰ r))
    -/
    induction' x using OreLocalization.ind with r₁ s₁
    /-
      case c
      R : Type u_1
      inst✝⁵ : Semiring R
      S : Submonoid R
      inst✝⁴ : OreLocalization.OreSet S
      X : Type u_2
      inst✝³ : AddCommMonoid X
      inst✝² : Module R X
      R₀ : Type ?u.30734
      inst✝¹ : CommSemiring R₀
      inst✝ : Algebra R₀ R
      r : R₀
      r₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul (__spread✝¹⁻⁰ r) (OreLocalization.oreDiv r₁ s₁)) (HMul.hMul (O …
    -/
    dsimp
    rw [mul_div_one, oreDiv_mul_char _ _ _ _ (algebraMap R₀ R r) s₁ (Algebra.commutes _ _).symm,
      Algebra.commutes, mul_one]
  smul_def' r x := by
    /-
      R : Type u_1
      inst✝⁵ : Semiring R
      S : Submonoid R
      inst✝⁴ : OreLocalization.OreSet S
      X : Type u_2
      inst✝³ : AddCommMonoid X
      inst✝² : Module R X
      R₀ : Type ?u.30734
      inst✝¹ : CommSemiring R₀
      inst✝ : Algebra R₀ R
      r : R₀
      x : OreLocalization S R
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (__spread✝¹⁻⁰ r) x)
    -/
    dsimp
    /-
      R : Type u_1
      inst✝⁵ : Semiring R
      S : Submonoid R
      inst✝⁴ : OreLocalization.OreSet S
      X : Type u_2
      inst✝³ : AddCommMonoid X
      inst✝² : Module R X
      R₀ : Type ?u.30734
      inst✝¹ : CommSemiring R₀
      inst✝ : Algebra R₀ R
      r : R₀
      x : OreLocalization S R
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul (OreLocalization.oreDiv ((algebraMap R₀ R) r …
    -/
    rw [Algebra.algebraMap_eq_smul_one, ← smul_eq_mul, smul_one_oreDiv_one_smul]
    /-
      🎉 no goals
    -/


/-- The universal lift from a ring homomorphism `f : R →+* T`, which maps elements in `S` to
units of `T`, to a ring homomorphism `R[S⁻¹] →+* T`. This extends the construction on
monoids. -/
def universalHom : R[S⁻¹] →+* T :=
  { universalMulHom f.toMonoidHom fS hf with
    map_zero' := by
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        ⊢ Eq ((↑__src✝).toFun 0) 0
      -/
      simp only [RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe]
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        ⊢ Eq ((OreLocalization.universalMulHom (↑f) fS ⋯) 0) 0
      -/
      rw [OreLocalization.zero_def, universalMulHom_apply]
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        ⊢ Eq (HMul.hMul (↑(Inv.inv (fS 1))) (↑f 0)) 0
      -/
      simp
      /-
        🎉 no goals
      -/
    map_add' := fun x y => by
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        x y : OreLocalization S R
        ⊢ Eq ((↑__src✝).toFun (HAdd.hAdd x y)) (HAdd.hAdd ((↑__src✝).toFun x) ((↑__src …
      -/
      simp only [RingHom.toMonoidHom_eq_coe, OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe]
      /-
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        x y : OreLocalization S R
        ⊢ Eq ((OreLocalization.universalMulHom (↑f) fS ⋯) (HAdd.hAdd x y)) (HAdd.hAdd  …
      -/
      induction' x with r₁ s₁
      /-
        case c
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        y : OreLocalization S R
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        ⊢ Eq ((OreLocalization.universalMulHom (↑f) fS ⋯) (HAdd.hAdd (OreLocalization. …
      -/
      induction' y with r₂ s₂
      /-
        case c.c
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        ⊢ Eq ((OreLocalization.universalMulHom (↑f) fS ⋯) (HAdd.hAdd (OreLocalization. …
      -/
      rcases oreDivAddChar' r₁ r₂ s₁ s₂ with ⟨r₃, s₃, h₃, h₃'⟩
      /-
        case c.c.mk.mk.intro
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        h₃' : Eq (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂  …
        ⊢ Eq ((OreLocalization.universalMulHom (↑f) fS ⋯) (HAdd.hAdd (OreLocalization. …
      -/
      rw [h₃']
      /-
        case c.c.mk.mk.intro
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        h₃' : Eq (HAdd.hAdd (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂  …
        ⊢ Eq ((OreLocalization.universalMulHom (↑f) fS ⋯) (OreLocalization.oreDiv (HAd …
      -/
      clear h₃'
      simp only [RingHom.toMonoidHom_eq_coe, smul_eq_mul, universalMulHom_apply, MonoidHom.coe_coe,
        Submonoid.smul_def]
      /-
        case c.c.mk.mk.intro
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HMul.hMul (↑(Inv.inv (fS (HMul.hMul s₃ s₁)))) (f (HAdd.hAdd (HMul.hMul ( …
      -/
      simp only [mul_inv_rev, MonoidHom.map_mul, RingHom.map_add, RingHom.map_mul, Units.val_mul]
      /-
        case c.c.mk.mk.intro
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HMul.hMul (HMul.hMul ↑(Inv.inv (fS s₁)) ↑(Inv.inv (fS s₃))) (HAdd.hAdd ( …
      -/
      rw [mul_add, mul_assoc, ← mul_assoc _ (f s₃), hf, ← Units.val_mul]
      /-
        case c.c.mk.mk.intro
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HAdd.hAdd (HMul.hMul (↑(Inv.inv (fS s₁))) (HMul.hMul (↑(HMul.hMul (Inv.i …
      -/
      simp only [one_mul, inv_mul_cancel, Units.val_one]
      /-
        case c.c.mk.mk.intro
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HAdd.hAdd (HMul.hMul (↑(Inv.inv (fS s₁))) (f r₁)) (HMul.hMul (HMul.hMul  …
      -/
      congr 1
      /-
        case c.c.mk.mk.intro.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HMul.hMul (HMul.hMul ↑(Inv.inv (fS s₁)) ↑(Inv.inv (fS s₃))) (HMul.hMul ( …
      -/
      rw [← mul_assoc]
      /-
        case c.c.mk.mk.intro.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑(Inv.inv (fS s₁)) ↑(Inv.inv (fS s₃))) ( …
      -/
      congr 1
      /-
        case c.c.mk.mk.intro.e_a.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (HMul.hMul ↑s₃ ↑s₁) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HMul.hMul (HMul.hMul ↑(Inv.inv (fS s₁)) ↑(Inv.inv (fS s₃))) (f r₃)) ↑(In …
      -/
      norm_cast at h₃
      /-
        case c.c.mk.mk.intro.e_a.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (↑(HMul.hMul s₃ s₁)) (HMul.hMul r₃ ↑s₂)
        ⊢ Eq (HMul.hMul (HMul.hMul ↑(Inv.inv (fS s₁)) ↑(Inv.inv (fS s₃))) (f r₃)) ↑(In …
      -/
      have h₃' := Subtype.coe_eq_of_eq_mk h₃
      /-
        case c.c.mk.mk.intro.e_a.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (↑(HMul.hMul s₃ s₁)) (HMul.hMul r₃ ↑s₂)
        h₃' : Eq (HMul.hMul s₃ s₁) ⟨HMul.hMul r₃ ↑s₂, ⋯⟩
        ⊢ Eq (HMul.hMul (HMul.hMul ↑(Inv.inv (fS s₁)) ↑(Inv.inv (fS s₃))) (f r₃)) ↑(In …
      -/
      rw [← Units.val_mul, ← mul_inv_rev, ← fS.map_mul, h₃']
      /-
        case c.c.mk.mk.intro.e_a.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (↑(HMul.hMul s₃ s₁)) (HMul.hMul r₃ ↑s₂)
        h₃' : Eq (HMul.hMul s₃ s₁) ⟨HMul.hMul r₃ ↑s₂, ⋯⟩
        ⊢ Eq (HMul.hMul (↑(Inv.inv (fS ⟨HMul.hMul r₃ ↑s₂, ⋯⟩))) (f r₃)) ↑(Inv.inv (fS  …
      -/
      rw [Units.inv_mul_eq_iff_eq_mul, Units.eq_mul_inv_iff_mul_eq, ← hf, ← hf]
      /-
        case c.c.mk.mk.intro.e_a.e_a
        R : Type u_1
        inst✝⁴ : Semiring R
        S : Submonoid R
        inst✝³ : OreLocalization.OreSet S
        X : Type u_2
        inst✝² : AddCommMonoid X
        inst✝¹ : Module R X
        T : Type u_3
        inst✝ : Semiring T
        f : RingHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        r₁ : R
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₃ : Eq (↑(HMul.hMul s₃ s₁)) (HMul.hMul r₃ ↑s₂)
        h₃' : Eq (HMul.hMul s₃ s₁) ⟨HMul.hMul r₃ ↑s₂, ⋯⟩
        ⊢ Eq (HMul.hMul (f r₃) (f ↑s₂)) (f ↑⟨HMul.hMul r₃ ↑s₂, ⋯⟩)
      -/
      simp only [map_mul] }
      /-
        🎉 no goals
      -/


theorem universalHom_apply {r : R} {s : S} :
    universalHom f fS hf (r /ₒ s) = ((fS s)⁻¹ : Units T) * f r :=
  rfl


theorem universalHom_commutes {r : R} : universalHom f fS hf (numeratorHom r) = f r := by
  /-
    R : Type u_1
    inst✝² : Semiring R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    T : Type u_3
    inst✝ : Semiring T
    f : RingHom R T
    fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
    hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
    r : R
    ⊢ Eq ((OreLocalization.universalHom f fS hf) (OreLocalization.numeratorHom r)) …
  -/
  simp [numeratorHom_apply, universalHom_apply]
  /-
    🎉 no goals
  -/


theorem universalHom_unique (φ : R[S⁻¹] →+* T) (huniv : ∀ r : R, φ (numeratorHom r) = f r) :
    φ = universalHom f fS hf :=
  RingHom.coe_monoidHom_injective <| universalMulHom_unique (RingHom.toMonoidHom f) fS hf (↑φ) huniv


instance : Ring R[S⁻¹] where
  __ := inferInstanceAs (Semiring R[S⁻¹])
  __ := inferInstanceAs (AddGroup R[S⁻¹])


@[simp]
lemma zsmul_eq_zsmul (n : ℤ) (x : X[S⁻¹]) :
    letI inst := OreLocalization.instModuleOfIsScalarTower (R₀ := ℤ) (R := R) (X := X) (S := S)
    HSMul.hSMul (self := @instHSMul _ _ inst.toSMul) n x = n • x := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommGroup X
    inst✝ : Module R X
    n : Int
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul n x) (HSMul.hSMul n x)
  -/
  letI inst := OreLocalization.instModuleOfIsScalarTower (R₀ := ℤ) (R := R) (X := X) (S := S)
  /-
    R : Type u_1
    inst✝³ : Ring R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : AddCommGroup X
    inst✝ : Module R X
    n : Int
    x : OreLocalization S X
    inst : Module Int (OreLocalization S X) := OreLocalization.instModuleOfIsScala …
    ⊢ Eq (HSMul.hSMul n x) (HSMul.hSMul n x)
  -/
  exact congr($(AddCommGroup.uniqueIntModule.2 inst).smul n x)
  /-
    🎉 no goals
  -/


theorem numeratorHom_inj (hS : S ≤ nonZeroDivisorsRight R) :
    Function.Injective (numeratorHom : R → R[S⁻¹]) :=
  fun r₁ r₂ h => by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    hS : LE.le S (nonZeroDivisorsRight R)
    r₁ r₂ : R
    h : Eq (OreLocalization.numeratorHom r₁) (OreLocalization.numeratorHom r₂)
    ⊢ Eq r₁ r₂
  -/
  rw [numeratorHom_apply, numeratorHom_apply, oreDiv_eq_iff] at h
  /-
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    hS : LE.le S (nonZeroDivisorsRight R)
    r₁ r₂ : R
    h : Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u r₂) (HSMul.hSMul v  …
    ⊢ Eq r₁ r₂
  -/
  rcases h with ⟨u, v, h₁, h₂⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    hS : LE.le S (nonZeroDivisorsRight R)
    r₁ r₂ : R
    u : Subtype fun x => Membership.mem S x
    v : R
    h₁ : Eq (HSMul.hSMul u r₂) (HSMul.hSMul v r₁)
    h₂ : Eq (HMul.hMul ↑u ↑1) (HMul.hMul v ↑1)
    ⊢ Eq r₁ r₂
  -/
  simp only [S.coe_one, mul_one, Submonoid.smul_def, smul_eq_mul] at h₁ h₂
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    hS : LE.le S (nonZeroDivisorsRight R)
    r₁ r₂ : R
    u : Subtype fun x => Membership.mem S x
    v : R
    h₁ : Eq (HMul.hMul (↑u) r₂) (HMul.hMul v r₁)
    h₂ : Eq (↑u) v
    ⊢ Eq r₁ r₂
  -/
  rw [← h₂, ← sub_eq_zero, ← mul_sub] at h₁
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    hS : LE.le S (nonZeroDivisorsRight R)
    r₁ r₂ : R
    u : Subtype fun x => Membership.mem S x
    v : R
    h₁ : Eq (HMul.hMul (↑u) (HSub.hSub r₂ r₁)) 0
    h₂ : Eq (↑u) v
    ⊢ Eq r₁ r₂
  -/
  exact (sub_eq_zero.mp (hS u.2 _ h₁)).symm
  /-
    🎉 no goals
  -/


theorem subsingleton_iff :
    Subsingleton R[S⁻¹] ↔ 0 ∈ S := by
  rw [← subsingleton_iff_zero_eq_one, OreLocalization.one_def,
    OreLocalization.zero_def, oreDiv_eq_iff]
  /-
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    ⊢ Iff (Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u 1) (HSMul.hSMul  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nontrivial_iff :
    Nontrivial R[S⁻¹] ↔ 0 ∉ S := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    ⊢ Iff (Nontrivial (OreLocalization S R)) (Not (Membership.mem S 0))
  -/
  rw [← not_subsingleton_iff_nontrivial, subsingleton_iff]
  /-
    🎉 no goals
  -/


theorem nontrivial_of_nonZeroDivisors [Nontrivial R] (hS : S ≤ R⁰) :
    Nontrivial R[S⁻¹] :=
  nontrivial_iff.mpr (fun e ↦ one_ne_zero <| hS e 1 (mul_zero _))


instance nontrivial : Nontrivial R[R⁰⁻¹] :=
  nontrivial_of_nonZeroDivisors (refl R⁰)


open Classical in
/-- The inversion of Ore fractions for a ring without zero divisors, satisfying `0⁻¹ = 0` and
`(r /ₒ r')⁻¹ = r' /ₒ r` for `r ≠ 0`. -/
@[irreducible]
protected def inv : R[R⁰⁻¹] → R[R⁰⁻¹] :=
  liftExpand
    (fun r s =>
      if hr : r = (0 : R) then (0 : R[R⁰⁻¹])
      else s /ₒ ⟨r, fun _ => eq_zero_of_ne_zero_of_mul_right_eq_zero hr⟩)
    (by
      /-
        R : Type u_1
        inst✝³ : Ring R
        inst✝² : Nontrivial R
        inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
        inst✝ : NoZeroDivisors R
        ⊢ ∀ (r t : R) (s : Subtype fun x => Membership.mem (nonZeroDivisors R) x) (ht  …
      -/
      intro r t s hst
      /-
        R : Type u_1
        inst✝³ : Ring R
        inst✝² : Nontrivial R
        inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
        inst✝ : NoZeroDivisors R
        r t : R
        s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
        hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
        ⊢ Eq ((fun r s => dite (Eq r 0) (fun hr => 0) fun hr => OreLocalization.oreDiv …
      -/
      by_cases hr : r = 0
        /-
          case pos
          R : Type u_1
          inst✝³ : Ring R
          inst✝² : Nontrivial R
          inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
          inst✝ : NoZeroDivisors R
          r t : R
          s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
          hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
          hr : Eq r 0
          ⊢ Eq ((fun r s => dite (Eq r 0) (fun hr => 0) fun hr => OreLocalization.oreDiv …
        -/
      · simp [hr]
        /-
          🎉 no goals
        -/
        /-
          case neg
          R : Type u_1
          inst✝³ : Ring R
          inst✝² : Nontrivial R
          inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
          inst✝ : NoZeroDivisors R
          r t : R
          s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
          hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
          hr : Not (Eq r 0)
          ⊢ Eq ((fun r s => dite (Eq r 0) (fun hr => 0) fun hr => OreLocalization.oreDiv …
        -/
      · by_cases ht : t = 0
          /-
            case pos
            R : Type u_1
            inst✝³ : Ring R
            inst✝² : Nontrivial R
            inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
            inst✝ : NoZeroDivisors R
            r t : R
            s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
            hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
            hr : Not (Eq r 0)
            ht : Eq t 0
            ⊢ Eq ((fun r s => dite (Eq r 0) (fun hr => 0) fun hr => OreLocalization.oreDiv …
          -/
        · exfalso
          /-
            case pos
            R : Type u_1
            inst✝³ : Ring R
            inst✝² : Nontrivial R
            inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
            inst✝ : NoZeroDivisors R
            r t : R
            s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
            hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
            hr : Not (Eq r 0)
            ht : Eq t 0
            ⊢ False
          -/
          apply nonZeroDivisors.coe_ne_zero ⟨_, hst⟩
          /-
            case pos
            R : Type u_1
            inst✝³ : Ring R
            inst✝² : Nontrivial R
            inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
            inst✝ : NoZeroDivisors R
            r t : R
            s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
            hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
            hr : Not (Eq r 0)
            ht : Eq t 0
            ⊢ Eq (↑⟨HMul.hMul t ↑s, hst⟩) 0
          -/
          simp [ht, mul_zero]
          /-
            🎉 no goals
          -/
          /-
            case neg
            R : Type u_1
            inst✝³ : Ring R
            inst✝² : Nontrivial R
            inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
            inst✝ : NoZeroDivisors R
            r t : R
            s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
            hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
            hr : Not (Eq r 0)
            ht : Not (Eq t 0)
            ⊢ Eq ((fun r s => dite (Eq r 0) (fun hr => 0) fun hr => OreLocalization.oreDiv …
          -/
        · simp only [hr, ht, dif_neg, not_false_iff, or_self_iff, mul_eq_zero, smul_eq_mul]
          /-
            case neg
            R : Type u_1
            inst✝³ : Ring R
            inst✝² : Nontrivial R
            inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
            inst✝ : NoZeroDivisors R
            r t : R
            s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
            hst : Membership.mem (nonZeroDivisors R) (HMul.hMul t ↑s)
            hr : Not (Eq r 0)
            ht : Not (Eq t 0)
            ⊢ Eq (OreLocalization.oreDiv ↑s ⟨r, ⋯⟩) (OreLocalization.oreDiv (HMul.hMul t ↑ …
          -/
          apply OreLocalization.expand)
          /-
            🎉 no goals
          -/


instance inv' : Inv R[R⁰⁻¹] :=
  ⟨OreLocalization.inv⟩


open Classical in
protected theorem inv_def {r : R} {s : R⁰} :
    (r /ₒ s)⁻¹ =
      if hr : r = (0 : R) then (0 : R[R⁰⁻¹])
      else s /ₒ ⟨r, fun _ => eq_zero_of_ne_zero_of_mul_right_eq_zero hr⟩ := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    ⊢ Eq (Inv.inv (OreLocalization.oreDiv r s)) (dite (Eq r 0) (fun hr => 0) fun h …
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


protected theorem mul_inv_cancel (x : R[R⁰⁻¹]) (h : x ≠ 0) : x * x⁻¹ = 1 := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    x : OreLocalization (nonZeroDivisors R) R
    h : Ne x 0
    ⊢ Eq (HMul.hMul x (Inv.inv x)) 1
  -/
  induction' x with r s
  /-
    case c
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    h : Ne (OreLocalization.oreDiv r s) 0
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r s) (Inv.inv (OreLocalization.oreDiv  …
  -/
  rw [OreLocalization.inv_def, OreLocalization.one_def]
  have hr : r ≠ 0 := by
    rintro rfl
    simp at h
  /-
    case c
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    h : Ne (OreLocalization.oreDiv r s) 0
    hr : Ne r 0
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r s) (dite (Eq r 0) (fun hr => 0) fun  …
  -/
  simp only [hr]
  /-
    case c
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    r : R
    s : Subtype fun x => Membership.mem (nonZeroDivisors R) x
    h : Ne (OreLocalization.oreDiv r s) 0
    hr : Ne r 0
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r s) (dite False (fun h => 0) fun h => …
  -/
  with_unfolding_all apply OreLocalization.mul_inv ⟨r, _⟩
  /-
    🎉 no goals
  -/


protected theorem inv_zero : (0 : R[R⁰⁻¹])⁻¹ = 0 := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    ⊢ Eq (Inv.inv 0) 0
  -/
  rw [OreLocalization.zero_def, OreLocalization.inv_def]
  /-
    R : Type u_1
    inst✝³ : Ring R
    inst✝² : Nontrivial R
    inst✝¹ : OreLocalization.OreSet (nonZeroDivisors R)
    inst✝ : NoZeroDivisors R
    ⊢ Eq (dite (Eq 0 0) (fun hr => 0) fun hr => OreLocalization.oreDiv ↑1 ⟨0, ⋯⟩)  …
  -/
  simp
  /-
    🎉 no goals
  -/


instance : DivisionRing R[R⁰⁻¹] where
  mul_inv_cancel := OreLocalization.mul_inv_cancel
  inv_zero := OreLocalization.inv_zero
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


instance : CommSemiring R[S⁻¹] where
  __ := inferInstanceAs (Semiring R[S⁻¹])
  __ := inferInstanceAs (CommMonoid R[S⁻¹])


instance : CommRing R[S⁻¹] where
  __ := inferInstanceAs (Ring R[S⁻¹])
  __ := inferInstanceAs (CommMonoid R[S⁻¹])


noncomputable
instance : Field R[R⁰⁻¹] where
  __ := inferInstanceAs (DivisionRing R[R⁰⁻¹])
  __ := inferInstanceAs (CommMonoid R[R⁰⁻¹])


