lemma ConvexOn.smul' (hf : ConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x)
    (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : MonovaryOn f g s) : ConvexOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConvexOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : MonovaryOn f g s
    ⊢ ConvexOn 𝕜 s (HSMul.hSMul f g)
  -/
  refine ⟨hf.1, fun x hx y hy a b ha hb hab ↦ ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConvexOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : MonovaryOn f g s
    x : 𝕜
    hx : Membership.mem s x
    y : 𝕜
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (HSMul.hSMul f g (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAd …
  -/
  dsimp
  refine
    (smul_le_smul (hf.2 hx hy ha hb hab) (hg.2 hx hy ha hb hab) (hf₀ <| hf.1 hx hy ha hb hab) <|
      add_nonneg (smul_nonneg ha <| hg₀ hx) <| smul_nonneg hb <| hg₀ hy).trans ?_
  calc
      _ = (a * a) • (f x • g x) + (b * b) • (f y • g y) + (a * b) • (f x • g y + f y • g x) := ?_
    _ ≤ (a * a) • (f x • g x) + (b * b) • (f y • g y) + (a * b) • (f x • g x + f y • g y) := by
        gcongr _ + (a * b) • ?_; exact hfg.smul_add_smul_le_smul_add_smul hx hy
    _ = (a * (a + b)) • (f x • g x) + (b * (a + b)) • (f y • g y) := by
        simp only [mul_add, add_smul, smul_add, mul_comm _ a]; abel
    _ = _ := by simp_rw [hab, mul_one]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConvexOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : MonovaryOn f g s
    x : 𝕜
    hx : Membership.mem s x
    y : 𝕜
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd (HSMul.hSMul a (f x)) (HSMul.hSMul b (f y))) (HAd …
  -/
  simp only [mul_add, add_smul, smul_add]
  rw [← smul_smul_smul_comm a, ← smul_smul_smul_comm b, ← smul_smul_smul_comm a b,
    ← smul_smul_smul_comm b b, smul_eq_mul, smul_eq_mul, smul_eq_mul, smul_eq_mul, mul_comm b,
    add_comm _ ((b * b) • f y • g y), add_add_add_comm, add_comm ((a * b) • f y • g x)]


lemma ConcaveOn.smul' [OrderedSMul 𝕜 E] (hf : ConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConcaveOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : AntivaryOn f g s
    ⊢ ConcaveOn 𝕜 s (HSMul.hSMul f g)
  -/
  refine ⟨hf.1, fun x hx y hy a b ha hb hab ↦ ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConcaveOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : AntivaryOn f g s
    x : 𝕜
    hx : Membership.mem s x
    y : 𝕜
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul a (HSMul.hSMul f g x)) (HSMul.hSMul b (HSMul.h …
  -/
  dsimp
  refine (smul_le_smul (hf.2 hx hy ha hb hab) (hg.2 hx hy ha hb hab)
    (add_nonneg (smul_nonneg ha <| hf₀ hx) <| smul_nonneg hb <| hf₀ hy)
    (hg₀ <| hf.1 hx hy ha hb hab)).trans' ?_
  calc a • f x • g x + b • f y • g y
        = (a * (a + b)) • (f x • g x) + (b * (a + b)) • (f y • g y) := by simp_rw [hab, mul_one]
    _ = (a * a) • (f x • g x) + (b * b) • (f y • g y) + (a * b) • (f x • g x + f y • g y) := by
        simp only [mul_add, add_smul, smul_add, mul_comm _ a]; abel
    _ ≤ (a * a) • (f x • g x) + (b * b) • (f y • g y) + (a * b) • (f x • g y + f y • g x) := by
        gcongr _ + (a * b) • ?_; exact hfg.smul_add_smul_le_smul_add_smul hx hy
    _ = _ := ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConcaveOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : AntivaryOn f g s
    x : 𝕜
    hx : Membership.mem s x
    y : 𝕜
    hy : Membership.mem s y
    a b : 𝕜
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul (HMul.hMul a a) (HSMul.hSMul (f x) (g  …
  -/
  simp only [mul_add, add_smul, smul_add]
  rw [← smul_smul_smul_comm a, ← smul_smul_smul_comm b, ← smul_smul_smul_comm a b,
    ← smul_smul_smul_comm b b, smul_eq_mul, smul_eq_mul, smul_eq_mul, smul_eq_mul, mul_comm b a,
    add_comm ((a * b) • f x • g y), add_comm ((a * b) • f x • g y), add_add_add_comm]


lemma ConvexOn.smul'' [OrderedSMul 𝕜 E] (hf : ConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0) (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConvexOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (f x) 0
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (g x) 0
    hfg : AntivaryOn f g s
    ⊢ ConcaveOn 𝕜 s (HSMul.hSMul f g)
  -/
  rw [← neg_smul_neg]
  exact hf.neg.smul' hg.neg (fun x hx ↦ neg_nonneg.2 <| hf₀ hx) (fun x hx ↦ neg_nonneg.2 <| hg₀ hx)
    hfg.neg


lemma ConcaveOn.smul'' (hf : ConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0)
    (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : MonovaryOn f g s) : ConvexOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConcaveOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (f x) 0
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (g x) 0
    hfg : MonovaryOn f g s
    ⊢ ConvexOn 𝕜 s (HSMul.hSMul f g)
  -/
  rw [← neg_smul_neg]
  exact hf.neg.smul' hg.neg (fun x hx ↦ neg_nonneg.2 <| hf₀ hx) (fun x hx ↦ neg_nonneg.2 <| hg₀ hx)
    hfg.neg


lemma ConvexOn.smul_concaveOn (hf : ConvexOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConvexOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (g x) 0
    hfg : AntivaryOn f g s
    ⊢ ConcaveOn 𝕜 s (HSMul.hSMul f g)
  -/
  rw [← neg_convexOn_iff, ← smul_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConvexOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (g x) 0
    hfg : AntivaryOn f g s
    ⊢ ConvexOn 𝕜 s (HSMul.hSMul f (Neg.neg g))
  -/
  exact hf.smul' hg.neg hf₀ (fun x hx ↦ neg_nonneg.2 <| hg₀ hx) hfg.neg_right
  /-
    🎉 no goals
  -/


lemma ConcaveOn.smul_convexOn [OrderedSMul 𝕜 E] (hf : ConcaveOn 𝕜 s f) (hg : ConvexOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : MonovaryOn f g s) :
    ConvexOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConcaveOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (g x) 0
    hfg : MonovaryOn f g s
    ⊢ ConvexOn 𝕜 s (HSMul.hSMul f g)
  -/
  rw [← neg_concaveOn_iff, ← smul_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConcaveOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (g x) 0
    hfg : MonovaryOn f g s
    ⊢ ConcaveOn 𝕜 s (HSMul.hSMul f (Neg.neg g))
  -/
  exact hf.smul' hg.neg hf₀ (fun x hx ↦ neg_nonneg.2 <| hg₀ hx) hfg.neg_right
  /-
    🎉 no goals
  -/


lemma ConvexOn.smul_concaveOn' [OrderedSMul 𝕜 E] (hf : ConvexOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0) (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : MonovaryOn f g s) :
    ConvexOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConvexOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (f x) 0
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : MonovaryOn f g s
    ⊢ ConvexOn 𝕜 s (HSMul.hSMul f g)
  -/
  rw [← neg_concaveOn_iff, ← smul_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹⁰ : LinearOrderedCommRing 𝕜
    inst✝⁹ : LinearOrderedCommRing E
    inst✝⁸ : LinearOrderedAddCommGroup F
    inst✝⁷ : Module 𝕜 E
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : Module E F
    inst✝⁴ : IsScalarTower 𝕜 E F
    inst✝³ : SMulCommClass 𝕜 E F
    inst✝² : OrderedSMul 𝕜 F
    inst✝¹ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    inst✝ : OrderedSMul 𝕜 E
    hf : ConvexOn 𝕜 s f
    hg : ConcaveOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (f x) 0
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : MonovaryOn f g s
    ⊢ ConcaveOn 𝕜 s (HSMul.hSMul f (Neg.neg g))
  -/
  exact hf.smul'' hg.neg hf₀ (fun x hx ↦ neg_nonpos.2 <| hg₀ hx) hfg.neg_right
  /-
    🎉 no goals
  -/


lemma ConcaveOn.smul_convexOn' (hf : ConcaveOn 𝕜 s f) (hg : ConvexOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0) (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f • g) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConcaveOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (f x) 0
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : AntivaryOn f g s
    ⊢ ConcaveOn 𝕜 s (HSMul.hSMul f g)
  -/
  rw [← neg_convexOn_iff, ← smul_neg]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : LinearOrderedCommRing 𝕜
    inst✝⁸ : LinearOrderedCommRing E
    inst✝⁷ : LinearOrderedAddCommGroup F
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : Module 𝕜 F
    inst✝⁴ : Module E F
    inst✝³ : IsScalarTower 𝕜 E F
    inst✝² : SMulCommClass 𝕜 E F
    inst✝¹ : OrderedSMul 𝕜 F
    inst✝ : OrderedSMul E F
    s : Set 𝕜
    f : 𝕜 → E
    g : 𝕜 → F
    hf : ConcaveOn 𝕜 s f
    hg : ConvexOn 𝕜 s g
    hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le (f x) 0
    hg₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (g x)
    hfg : AntivaryOn f g s
    ⊢ ConvexOn 𝕜 s (HSMul.hSMul f (Neg.neg g))
  -/
  exact hf.smul'' hg.neg hf₀ (fun x hx ↦ neg_nonpos.2 <| hg₀ hx) hfg.neg_right
  /-
    🎉 no goals
  -/


lemma ConvexOn.mul (hf : ConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x)
    (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : MonovaryOn f g s) :
    ConvexOn 𝕜 s (f * g) := hf.smul' hg hf₀ hg₀ hfg


lemma ConcaveOn.mul (hf : ConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f * g) := hf.smul' hg hf₀ hg₀ hfg


lemma ConvexOn.mul' (hf : ConvexOn 𝕜 s f) (hg : ConvexOn 𝕜 s g) (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0)
    (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f * g) := hf.smul'' hg hf₀ hg₀ hfg


lemma ConcaveOn.mul' (hf : ConcaveOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g) (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0)
    (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : MonovaryOn f g s) :
    ConvexOn 𝕜 s (f * g) := hf.smul'' hg hf₀ hg₀ hfg


lemma ConvexOn.mul_concaveOn (hf : ConvexOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f * g) := hf.smul_concaveOn hg hf₀ hg₀ hfg


lemma ConcaveOn.mul_convexOn (hf : ConcaveOn 𝕜 s f) (hg : ConvexOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) (hg₀ : ∀ ⦃x⦄, x ∈ s → g x ≤ 0) (hfg : MonovaryOn f g s) :
    ConvexOn 𝕜 s (f * g) := hf.smul_convexOn hg hf₀ hg₀ hfg


lemma ConvexOn.mul_concaveOn' (hf : ConvexOn 𝕜 s f) (hg : ConcaveOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0) (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : MonovaryOn f g s) :
    ConvexOn 𝕜 s (f * g) := hf.smul_concaveOn' hg hf₀ hg₀ hfg


lemma ConcaveOn.mul_convexOn' (hf : ConcaveOn 𝕜 s f) (hg : ConvexOn 𝕜 s g)
    (hf₀ : ∀ ⦃x⦄, x ∈ s → f x ≤ 0) (hg₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ g x) (hfg : AntivaryOn f g s) :
    ConcaveOn 𝕜 s (f • g) := hf.smul_convexOn' hg hf₀ hg₀ hfg


lemma ConvexOn.pow (hf : ConvexOn 𝕜 s f) (hf₀ : ∀ ⦃x⦄, x ∈ s → 0 ≤ f x) :
    ∀ n, ConvexOn 𝕜 s (f ^ n)
            /-
              𝕜 : Type u_1
              E : Type u_2
              inst✝⁵ : LinearOrderedCommRing 𝕜
              inst✝⁴ : LinearOrderedCommRing E
              inst✝³ : Module 𝕜 E
              s : Set 𝕜
              inst✝² : OrderedSMul 𝕜 E
              inst✝¹ : IsScalarTower 𝕜 E E
              inst✝ : SMulCommClass 𝕜 E E
              f : 𝕜 → E
              hf : ConvexOn 𝕜 s f
              hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
              ⊢ ConvexOn 𝕜 s (HPow.hPow f 0)
            -/
  | 0 => by simpa using convexOn_const 1 hf.1
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁵ : LinearOrderedCommRing 𝕜
      inst✝⁴ : LinearOrderedCommRing E
      inst✝³ : Module 𝕜 E
      s : Set 𝕜
      inst✝² : OrderedSMul 𝕜 E
      inst✝¹ : IsScalarTower 𝕜 E E
      inst✝ : SMulCommClass 𝕜 E E
      f : 𝕜 → E
      hf : ConvexOn 𝕜 s f
      hf₀ : ∀ ⦃x : 𝕜⦄, Membership.mem s x → LE.le 0 (f x)
      n : Nat
      ⊢ ConvexOn 𝕜 s (HPow.hPow f (HAdd.hAdd n 1))
    -/
    rw [pow_succ']
    exact hf.mul (hf.pow hf₀ _) hf₀ (fun x hx ↦ pow_nonneg (hf₀ hx) _) <|
      (monovaryOn_self f s).pow_right₀ hf₀ n


/-- `x^n`, `n : ℕ` is convex on `[0, +∞)` for all `n`. -/
lemma convexOn_pow : ∀ n, ConvexOn 𝕜 (Ici 0) fun x : 𝕜 ↦ x ^ n :=
  (convexOn_id <| convex_Ici _).pow fun _ ↦ id


/-- `x^n`, `n : ℕ` is convex on the whole real line whenever `n` is even. -/
protected lemma Even.convexOn_pow {n : ℕ} (hn : Even n) : ConvexOn 𝕜 univ fun x : 𝕜 ↦ x ^ n := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedCommRing 𝕜
    n : Nat
    hn : Even n
    ⊢ ConvexOn 𝕜 Set.univ fun x => HPow.hPow x n
  -/
  obtain ⟨n, rfl⟩ := hn
  /-
    case intro
    𝕜 : Type u_1
    inst✝ : LinearOrderedCommRing 𝕜
    n : Nat
    ⊢ ConvexOn 𝕜 Set.univ fun x => HPow.hPow x (HAdd.hAdd n n)
  -/
  simp_rw [← two_mul, pow_mul]
  refine ConvexOn.pow ⟨convex_univ, fun x _ y _ a b ha hb hab ↦ sub_nonneg.1 ?_⟩
    (fun _ _ ↦ by positivity) _
  calc
    (0 : 𝕜) ≤ (a * b) * (x - y) ^ 2 := by positivity
    _ = _ := by obtain rfl := eq_sub_of_add_eq hab; simp only [smul_eq_mul]; ring


open Int in
/-- `x^m`, `m : ℤ` is convex on `(0, +∞)` for all `m`. -/
lemma convexOn_zpow : ∀ n : ℤ, ConvexOn 𝕜 (Ioi 0) fun x : 𝕜 ↦ x ^ n
  | (n : ℕ) => by
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      ⊢ ConvexOn 𝕜 (Set.Ioi 0) fun x => HPow.hPow x ↑n
    -/
    simp_rw [zpow_natCast]
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      ⊢ ConvexOn 𝕜 (Set.Ioi 0) fun x => HPow.hPow x n
    -/
    exact (convexOn_pow n).subset Ioi_subset_Ici_self (convex_Ioi _)
    /-
      🎉 no goals
    -/
  | -[n+1] => by
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      ⊢ ConvexOn 𝕜 (Set.Ioi 0) fun x => HPow.hPow x (Int.negSucc n)
    -/
    simp_rw [zpow_negSucc, ← inv_pow]
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      ⊢ ConvexOn 𝕜 (Set.Ioi 0) fun x => HPow.hPow (Inv.inv x) (HAdd.hAdd n 1)
    -/
    refine (convexOn_iff_forall_pos.2 ⟨convex_Ioi _, ?_⟩).pow (fun x (hx : 0 < x) ↦ by positivity) _
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      ⊢ ∀ ⦃x : 𝕜⦄, Membership.mem (Set.Ioi 0) x → ∀ ⦃y : 𝕜⦄, Membership.mem (Set.Ioi …
    -/
    rintro x (hx : 0 < x) y (hy : 0 < y) a b ha hb hab
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      x : 𝕜
      hx : LT.lt 0 x
      y : 𝕜
      hy : LT.lt 0 y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (Inv.inv (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) (HAdd.hAdd ( …
    -/
    field_simp
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      x : 𝕜
      hx : LT.lt 0 x
      y : 𝕜
      hy : LT.lt 0 y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LE.le (HDiv.hDiv 1 (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))) (HDiv.hDiv ( …
    -/
    rw [div_le_div_iff₀, ← sub_nonneg]
    · calc
        0 ≤ a * b * (x - y) ^ 2 := by positivity
        _ = _ := by obtain rfl := eq_sub_of_add_eq hab; ring
    /-
      case hb
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      n : Nat
      x : 𝕜
      hx : LT.lt 0 x
      y : 𝕜
      hy : LT.lt 0 y
      a b : 𝕜
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      ⊢ LT.lt 0 (HAdd.hAdd (HMul.hMul a x) (HMul.hMul b y))
    -/
    all_goals positivity
    /-
      🎉 no goals
    -/


