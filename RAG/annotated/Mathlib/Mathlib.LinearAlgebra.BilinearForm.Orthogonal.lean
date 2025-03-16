/-- The proposition that two elements of a bilinear form space are orthogonal. For orthogonality
of an indexed set of elements, use `BilinForm.iIsOrtho`. -/
def IsOrtho (B : BilinForm R M) (x y : M) : Prop :=
  B x y = 0


theorem isOrtho_def {B : BilinForm R M} {x y : M} : B.IsOrtho x y ↔ B x y = 0 :=
  Iff.rfl


theorem isOrtho_zero_left (x : M) : IsOrtho B (0 : M) x := LinearMap.isOrtho_zero_left B x


theorem isOrtho_zero_right (x : M) : IsOrtho B x (0 : M) :=
  zero_right x


theorem ne_zero_of_not_isOrtho_self {B : BilinForm K V} (x : V) (hx₁ : ¬B.IsOrtho x x) : x ≠ 0 :=
  fun hx₂ => hx₁ (hx₂.symm ▸ isOrtho_zero_left _)


theorem IsRefl.ortho_comm (H : B.IsRefl) {x y : M} : IsOrtho B x y ↔ IsOrtho B y x :=
  ⟨eq_zero H, eq_zero H⟩


theorem IsAlt.ortho_comm (H : B₁.IsAlt) {x y : M₁} : IsOrtho B₁ x y ↔ IsOrtho B₁ y x :=
  LinearMap.IsAlt.ortho_comm H


theorem IsSymm.ortho_comm (H : B.IsSymm) {x y : M} : IsOrtho B x y ↔ IsOrtho B y x :=
  LinearMap.IsSymm.ortho_comm H


/-- A set of vectors `v` is orthogonal with respect to some bilinear form `B` if and only
if for all `i ≠ j`, `B (v i) (v j) = 0`. For orthogonality between two elements, use
`BilinForm.IsOrtho` -/
def iIsOrtho {n : Type w} (B : BilinForm R M) (v : n → M) : Prop :=
  B.IsOrthoᵢ v


theorem iIsOrtho_def {n : Type w} {B : BilinForm R M} {v : n → M} :
    B.iIsOrtho v ↔ ∀ i j : n, i ≠ j → B (v i) (v j) = 0 :=
  Iff.rfl


@[simp]
theorem isOrtho_smul_left {x y : M₄} {a : R₄} (ha : a ≠ 0) :
    IsOrtho G (a • x) y ↔ IsOrtho G x y := by
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Iff (G.IsOrtho (HSMul.hSMul a x) y) (G.IsOrtho x y)
  -/
  dsimp only [IsOrtho]
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Iff (Eq ((G (HSMul.hSMul a x)) y) 0) (Eq ((G x) y) 0)
  -/
  rw [map_smul]
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Iff (Eq ((HSMul.hSMul a (G x)) y) 0) (Eq ((G x) y) 0)
  -/
  simp only [LinearMap.smul_apply, smul_eq_mul, mul_eq_zero, or_iff_right_iff_imp]
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Eq a 0 → Eq ((G x) y) 0
  -/
  exact fun a ↦ (ha a).elim
  /-
    🎉 no goals
  -/


@[simp]
theorem isOrtho_smul_right {x y : M₄} {a : R₄} (ha : a ≠ 0) :
    IsOrtho G x (a • y) ↔ IsOrtho G x y := by
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Iff (G.IsOrtho x (HSMul.hSMul a y)) (G.IsOrtho x y)
  -/
  dsimp only [IsOrtho]
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Iff (Eq ((G x) (HSMul.hSMul a y)) 0) (Eq ((G x) y) 0)
  -/
  rw [map_smul]
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Iff (Eq (HSMul.hSMul a ((G x) y)) 0) (Eq ((G x) y) 0)
  -/
  simp only [smul_eq_mul, mul_eq_zero, or_iff_right_iff_imp]
  /-
    R₄ : Type u_7
    M₄ : Type u_8
    inst✝³ : CommRing R₄
    inst✝² : IsDomain R₄
    inst✝¹ : AddCommGroup M₄
    inst✝ : Module R₄ M₄
    G : LinearMap.BilinForm R₄ M₄
    x y : M₄
    a : R₄
    ha : Ne a 0
    ⊢ Eq a 0 → Eq ((G x) y) 0
  -/
  exact fun a ↦ (ha a).elim
  /-
    🎉 no goals
  -/


/-- A set of orthogonal vectors `v` with respect to some bilinear form `B` is linearly independent
  if for all `i`, `B (v i) (v i) ≠ 0`. -/
theorem linearIndependent_of_iIsOrtho {n : Type w} {B : BilinForm K V} {v : n → V}
    (hv₁ : B.iIsOrtho v) (hv₂ : ∀ i, ¬B.IsOrtho (v i) (v i)) : LinearIndependent K v := by
  classical
    rw [linearIndependent_iff']
    intro s w hs i hi
    have : B (s.sum fun i : n => w i • v i) (v i) = 0 := by rw [hs, zero_left]
    have hsum : (s.sum fun j : n => w j * B (v j) (v i)) = w i * B (v i) (v i) := by
      apply Finset.sum_eq_single_of_mem i hi
      intro j _ hij
      rw [iIsOrtho_def.1 hv₁ _ _ hij, mul_zero]
    simp_rw [sum_left, smul_left, hsum] at this
    exact eq_zero_of_ne_zero_of_mul_right_eq_zero (hv₂ i) this


/-- The orthogonal complement of a submodule `N` with respect to some bilinear form is the set of
elements `x` which are orthogonal to all elements of `N`; i.e., for all `y` in `N`, `B x y = 0`.

Note that for general (neither symmetric nor antisymmetric) bilinear forms this definition has a
chirality; in addition to this "left" orthogonal complement one could define a "right" orthogonal
complement for which, for all `y` in `N`, `B y x = 0`.  This variant definition is not currently
provided in mathlib. -/
def orthogonal (B : BilinForm R M) (N : Submodule R M) : Submodule R M where
  carrier := { m | ∀ n ∈ N, IsOrtho B n m }
  zero_mem' x _ := isOrtho_zero_right x
  add_mem' {x y} hx hy n hn := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      R₁ : Type u_3
      M₁ : Type u_4
      inst✝⁵ : CommRing R₁
      inst✝⁴ : AddCommGroup M₁
      inst✝³ : Module R₁ M₁
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B✝ : LinearMap.BilinForm R M
      B₁ : LinearMap.BilinForm R₁ M₁
      B : LinearMap.BilinForm R M
      N : Submodule R M
      x y : M
      hx : Membership.mem (setOf fun m => ∀ (n : M), Membership.mem N n → B.IsOrtho  …
      hy : Membership.mem (setOf fun m => ∀ (n : M), Membership.mem N n → B.IsOrtho  …
      n : M
      hn : Membership.mem N n
      ⊢ B.IsOrtho n (HAdd.hAdd x y)
    -/
    rw [IsOrtho, add_right, show B n x = 0 from hx n hn, show B n y = 0 from hy n hn, zero_add]
    /-
      🎉 no goals
    -/
  smul_mem' c x hx n hn := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁸ : CommSemiring R
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Module R M
      R₁ : Type u_3
      M₁ : Type u_4
      inst✝⁵ : CommRing R₁
      inst✝⁴ : AddCommGroup M₁
      inst✝³ : Module R₁ M₁
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B✝ : LinearMap.BilinForm R M
      B₁ : LinearMap.BilinForm R₁ M₁
      B : LinearMap.BilinForm R M
      N : Submodule R M
      c : R
      x : M
      hx : Membership.mem { carrier := setOf fun m => ∀ (n : M), Membership.mem N n  …
      n : M
      hn : Membership.mem N n
      ⊢ B.IsOrtho n (HSMul.hSMul c x)
    -/
    rw [IsOrtho, smul_right, show B n x = 0 from hx n hn, mul_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_orthogonal_iff {N : Submodule R M} {m : M} :
    m ∈ B.orthogonal N ↔ ∀ n ∈ N, IsOrtho B n m :=
  Iff.rfl


                                                        /-
                                                          R : Type u_1
                                                          M : Type u_2
                                                          inst✝² : CommSemiring R
                                                          inst✝¹ : AddCommMonoid M
                                                          inst✝ : Module R M
                                                          B : LinearMap.BilinForm R M
                                                          ⊢ Eq (B.orthogonal Bot.bot) Top.top
                                                        -/
@[simp] lemma orthogonal_bot : B.orthogonal ⊥ = ⊤ := by ext; simp [IsOrtho]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem orthogonal_le (h : N ≤ L) : B.orthogonal L ≤ B.orthogonal N := fun _ hn l hl => hn l (h hl)


theorem le_orthogonal_orthogonal (b : B.IsRefl) : N ≤ B.orthogonal (B.orthogonal N) :=
  fun n hn _ hm => b _ _ (hm n hn)


lemma orthogonal_top_eq_ker (hB : B.IsRefl) :
    B.orthogonal ⊤ = LinearMap.ker B := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    hB : B.IsRefl
    ⊢ Eq (B.orthogonal Top.top) (LinearMap.ker B)
  -/
  ext; simp [LinearMap.BilinForm.IsOrtho, LinearMap.ext_iff, hB.eq_iff]
       /-
         🎉 no goals
       -/


lemma orthogonal_top_eq_bot (hB : B.Nondegenerate) (hB₀ : B.IsRefl) :
    B.orthogonal ⊤ = ⊥ :=
  (Submodule.eq_bot_iff _).mpr fun _ hx ↦ hB _ fun y ↦ hB₀ _ _ <| hx y Submodule.mem_top

-- ↓ This lemma only applies in fields as we require `a * b = 0 → a = 0 ∨ b = 0`

theorem span_singleton_inf_orthogonal_eq_bot {B : BilinForm K V} {x : V} (hx : ¬B.IsOrtho x x) :
    (K ∙ x) ⊓ B.orthogonal (K ∙ x) = ⊥ := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Min.min (Submodule.span K (Singleton.singleton x)) (B.orthogonal (Submod …
  -/
  rw [← Finset.coe_singleton]
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.orthogonal (Submo …
  -/
  refine eq_bot_iff.2 fun y h => ?_
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    hx : Not (B.IsOrtho x x)
    y : V
    h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
    ⊢ Membership.mem Bot.bot y
  -/
  rcases mem_span_finset.1 h.1 with ⟨μ, rfl⟩
  /-
    case intro
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    hx : Not (B.IsOrtho x x)
    μ : V → K
    h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
    ⊢ Membership.mem Bot.bot ((Singleton.singleton x).sum fun i => HSMul.hSMul (μ  …
  -/
  have := h.2 x ?_
    /-
      case intro.refine_2
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      this : B.IsOrtho x ((Singleton.singleton x).sum fun i => HSMul.hSMul (μ i) i)
      ⊢ Membership.mem Bot.bot ((Singleton.singleton x).sum fun i => HSMul.hSMul (μ  …
    -/
  · rw [Finset.sum_singleton] at this ⊢
    /-
      case intro.refine_2
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      this : B.IsOrtho x (HSMul.hSMul (μ x) x)
      ⊢ Membership.mem Bot.bot (HSMul.hSMul (μ x) x)
    -/
    suffices hμzero : μ x = 0 by rw [hμzero, zero_smul, Submodule.mem_bot]
    /-
      case intro.refine_2
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      this : B.IsOrtho x (HSMul.hSMul (μ x) x)
      ⊢ Eq (μ x) 0
    -/
    change B x (μ x • x) = 0 at this
    /-
      case intro.refine_2
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      this : Eq ((B x) (HSMul.hSMul (μ x) x)) 0
      ⊢ Eq (μ x) 0
    -/
    rw [smul_right] at this
    /-
      case intro.refine_2
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      this : Eq (HMul.hMul (μ x) ((B x) x)) 0
      ⊢ Eq (μ x) 0
    -/
    exact eq_zero_of_ne_zero_of_mul_right_eq_zero hx this
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_1
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      ⊢ Membership.mem (Submodule.span K ↑(Singleton.singleton x)) x
    -/
  · rw [Submodule.mem_span]
    /-
      case intro.refine_1
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x : V
      hx : Not (B.IsOrtho x x)
      μ : V → K
      h : Membership.mem (Min.min (Submodule.span K ↑(Singleton.singleton x)) (B.ort …
      ⊢ ∀ (p : Submodule K V), HasSubset.Subset ↑(Singleton.singleton x) ↑p → Member …
    -/
    exact fun _ hp => hp <| Finset.mem_singleton_self _
    /-
      🎉 no goals
    -/

-- ↓ This lemma only applies in fields since we use the `mul_eq_zero`

theorem orthogonal_span_singleton_eq_toLin_ker {B : BilinForm K V} (x : V) :
    B.orthogonal (K ∙ x) = LinearMap.ker (LinearMap.BilinForm.toLinHomAux₁ B x) := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    ⊢ Eq (B.orthogonal (Submodule.span K (Singleton.singleton x))) (LinearMap.ker  …
  -/
  ext y
  /-
    case h
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x y : V
    ⊢ Iff (Membership.mem (B.orthogonal (Submodule.span K (Singleton.singleton x)) …
  -/
  simp_rw [mem_orthogonal_iff, LinearMap.mem_ker, Submodule.mem_span_singleton]
  /-
    case h
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x y : V
    ⊢ Iff (∀ (n : V), (Exists fun a => Eq (HSMul.hSMul a x) n) → B.IsOrtho n y) (E …
  -/
  constructor
    /-
      case h.mp
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x y : V
      ⊢ (∀ (n : V), (Exists fun a => Eq (HSMul.hSMul a x) n) → B.IsOrtho n y) → Eq ( …
    -/
  · exact fun h => h x ⟨1, one_smul _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x y : V
      ⊢ Eq ((B.toLinHomAux₁ x) y) 0 → ∀ (n : V), (Exists fun a => Eq (HSMul.hSMul a  …
    -/
  · rintro h _ ⟨z, rfl⟩
    /-
      case h.mpr.intro
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x y : V
      h : Eq ((B.toLinHomAux₁ x) y) 0
      z : K
      ⊢ B.IsOrtho (HSMul.hSMul z x) y
    -/
    rw [IsOrtho, smul_left, mul_eq_zero]
    /-
      case h.mpr.intro
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      x y : V
      h : Eq ((B.toLinHomAux₁ x) y) 0
      z : K
      ⊢ Or (Eq z 0) (Eq ((B x) y) 0)
    -/
    exact Or.intro_right _ h
    /-
      🎉 no goals
    -/


theorem span_singleton_sup_orthogonal_eq_top {B : BilinForm K V} {x : V} (hx : ¬B.IsOrtho x x) :
    (K ∙ x) ⊔ B.orthogonal (K ∙ x) = ⊤ := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Max.max (Submodule.span K (Singleton.singleton x)) (B.orthogonal (Submod …
  -/
  rw [orthogonal_span_singleton_eq_toLin_ker]
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ Eq (Max.max (Submodule.span K (Singleton.singleton x)) (LinearMap.ker (B.toL …
  -/
  exact LinearMap.span_singleton_sup_ker_eq_top _ hx
  /-
    🎉 no goals
  -/


/-- Given a bilinear form `B` and some `x` such that `B x x ≠ 0`, the span of the singleton of `x`
  is complement to its orthogonal complement. -/
theorem isCompl_span_singleton_orthogonal {B : BilinForm K V} {x : V} (hx : ¬B.IsOrtho x x) :
    IsCompl (K ∙ x) (B.orthogonal <| K ∙ x) :=
  { disjoint := disjoint_iff.2 <| span_singleton_inf_orthogonal_eq_bot hx
    codisjoint := codisjoint_iff.2 <| span_singleton_sup_orthogonal_eq_top hx }


/-- The restriction of a reflexive bilinear form `B` onto a submodule `W` is
nondegenerate if `Disjoint W (B.orthogonal W)`. -/
theorem nondegenerate_restrict_of_disjoint_orthogonal (B : BilinForm R₁ M₁) (b : B.IsRefl)
    {W : Submodule R₁ M₁} (hW : Disjoint W (B.orthogonal W)) : (B.restrict W).Nondegenerate := by
  /-
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.IsRefl
    W : Submodule R₁ M₁
    hW : Disjoint W (B.orthogonal W)
    ⊢ (B.restrict W).Nondegenerate
  -/
  rintro ⟨x, hx⟩ b₁
  /-
    case mk
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.IsRefl
    W : Submodule R₁ M₁
    hW : Disjoint W (B.orthogonal W)
    x : M₁
    hx : Membership.mem W x
    b₁ : ∀ (n : Subtype fun x => Membership.mem W x), Eq (((B.restrict W) ⟨x, hx⟩) …
    ⊢ Eq ⟨x, hx⟩ 0
  -/
  rw [Submodule.mk_eq_zero, ← Submodule.mem_bot R₁]
  /-
    case mk
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.IsRefl
    W : Submodule R₁ M₁
    hW : Disjoint W (B.orthogonal W)
    x : M₁
    hx : Membership.mem W x
    b₁ : ∀ (n : Subtype fun x => Membership.mem W x), Eq (((B.restrict W) ⟨x, hx⟩) …
    ⊢ Membership.mem Bot.bot x
  -/
  refine hW.le_bot ⟨hx, fun y hy => ?_⟩
  /-
    case mk
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.IsRefl
    W : Submodule R₁ M₁
    hW : Disjoint W (B.orthogonal W)
    x : M₁
    hx : Membership.mem W x
    b₁ : ∀ (n : Subtype fun x => Membership.mem W x), Eq (((B.restrict W) ⟨x, hx⟩) …
    y : M₁
    hy : Membership.mem W y
    ⊢ B.IsOrtho y x
  -/
  specialize b₁ ⟨y, hy⟩
  /-
    case mk
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.IsRefl
    W : Submodule R₁ M₁
    hW : Disjoint W (B.orthogonal W)
    x : M₁
    hx : Membership.mem W x
    y : M₁
    hy : Membership.mem W y
    b₁ : Eq (((B.restrict W) ⟨x, hx⟩) ⟨y, hy⟩) 0
    ⊢ B.IsOrtho y x
  -/
  simp only [restrict_apply, domRestrict_apply] at b₁
  /-
    case mk
    R₁ : Type u_3
    M₁ : Type u_4
    inst✝² : CommRing R₁
    inst✝¹ : AddCommGroup M₁
    inst✝ : Module R₁ M₁
    B : LinearMap.BilinForm R₁ M₁
    b : B.IsRefl
    W : Submodule R₁ M₁
    hW : Disjoint W (B.orthogonal W)
    x : M₁
    hx : Membership.mem W x
    y : M₁
    hy : Membership.mem W y
    b₁ : Eq ((B x) y) 0
    ⊢ B.IsOrtho y x
  -/
  exact isOrtho_def.mpr (b x y b₁)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-30")]
alias nondegenerateRestrictOfDisjointOrthogonal := nondegenerate_restrict_of_disjoint_orthogonal


/-- An orthogonal basis with respect to a nondegenerate bilinear form has no self-orthogonal
elements. -/
theorem iIsOrtho.not_isOrtho_basis_self_of_nondegenerate {n : Type w} [Nontrivial R]
    {B : BilinForm R M} {v : Basis n R M} (h : B.iIsOrtho v) (hB : B.Nondegenerate) (i : n) :
    ¬B.IsOrtho (v i) (v i) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ⊢ Not (B.IsOrtho (v i) (v i))
  -/
  intro ho
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    ⊢ False
  -/
  refine v.ne_zero i (hB (v i) fun m => ?_)
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    m : M
    ⊢ Eq ((B (v i)) m) 0
  -/
  obtain ⟨vi, rfl⟩ := v.repr.symm.surjective m
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    ⊢ Eq ((B (v i)) (v.repr.symm vi)) 0
  -/
  rw [Basis.repr_symm_apply, Finsupp.linearCombination_apply, Finsupp.sum, sum_right]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    ⊢ Eq (vi.support.sum fun i_1 => (B (v i)) (HSMul.hSMul (vi i_1) (v i_1))) 0
  -/
  apply Finset.sum_eq_zero
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    ⊢ ∀ (x : n), Membership.mem vi.support x → Eq ((B (v i)) (HSMul.hSMul (vi x) ( …
  -/
  rintro j -
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    j : n
    ⊢ Eq ((B (v i)) (HSMul.hSMul (vi j) (v j))) 0
  -/
  rw [smul_right]
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    j : n
    ⊢ Eq (HMul.hMul (vi j) ((B (v i)) (v j))) 0
  -/
  convert mul_zero (vi j) using 2
  /-
    case h.e'_2.h.e'_6
    R : Type u_1
    M : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    n : Type w
    inst✝ : Nontrivial R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    h : B.iIsOrtho ⇑v
    hB : B.Nondegenerate
    i : n
    ho : B.IsOrtho (v i) (v i)
    vi : Finsupp n R
    j : n
    ⊢ Eq ((B (v i)) (v j)) 0
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case h.e'_2.h.e'_6.inl
      R : Type u_1
      M : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      n : Type w
      inst✝ : Nontrivial R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      h : B.iIsOrtho ⇑v
      hB : B.Nondegenerate
      i : n
      ho : B.IsOrtho (v i) (v i)
      vi : Finsupp n R
      ⊢ Eq ((B (v i)) (v i)) 0
    -/
  · exact ho
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_6.inr
      R : Type u_1
      M : Type u_2
      inst✝³ : CommSemiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      n : Type w
      inst✝ : Nontrivial R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      h : B.iIsOrtho ⇑v
      hB : B.Nondegenerate
      i : n
      ho : B.IsOrtho (v i) (v i)
      vi : Finsupp n R
      j : n
      hij : Ne i j
      ⊢ Eq ((B (v i)) (v j)) 0
    -/
  · exact h hij
    /-
      🎉 no goals
    -/


/-- Given an orthogonal basis with respect to a bilinear form, the bilinear form is nondegenerate
iff the basis has no elements which are self-orthogonal. -/
theorem iIsOrtho.nondegenerate_iff_not_isOrtho_basis_self {n : Type w} [Nontrivial R]
    [NoZeroDivisors R] (B : BilinForm R M) (v : Basis n R M) (hO : B.iIsOrtho v) :
    B.Nondegenerate ↔ ∀ i, ¬B.IsOrtho (v i) (v i) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ⊢ Iff B.Nondegenerate (∀ (i : n), Not (B.IsOrtho (v i) (v i)))
  -/
  refine ⟨hO.not_isOrtho_basis_self_of_nondegenerate, fun ho m hB => ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    m : M
    hB : ∀ (n : M), Eq ((B m) n) 0
    ⊢ Eq m 0
  -/
  obtain ⟨vi, rfl⟩ := v.repr.symm.surjective m
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (n_1 : M), Eq ((B (v.repr.symm vi)) n_1) 0
    ⊢ Eq (v.repr.symm vi) 0
  -/
  rw [LinearEquiv.map_eq_zero_iff]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (n_1 : M), Eq ((B (v.repr.symm vi)) n_1) 0
    ⊢ Eq vi 0
  -/
  ext i
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (n_1 : M), Eq ((B (v.repr.symm vi)) n_1) 0
    i : n
    ⊢ Eq (vi i) (0 i)
  -/
  rw [Finsupp.zero_apply]
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    hB : ∀ (n_1 : M), Eq ((B (v.repr.symm vi)) n_1) 0
    i : n
    ⊢ Eq (vi i) 0
  -/
  specialize hB (v i)
  simp_rw [Basis.repr_symm_apply, Finsupp.linearCombination_apply, Finsupp.sum, sum_left,
           smul_left] at hB
  /-
    case intro.h
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommSemiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    n : Type w
    inst✝¹ : Nontrivial R
    inst✝ : NoZeroDivisors R
    B : LinearMap.BilinForm R M
    v : Basis n R M
    hO : B.iIsOrtho ⇑v
    ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
    vi : Finsupp n R
    i : n
    hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
    ⊢ Eq (vi i) 0
  -/
  rw [Finset.sum_eq_single i] at hB
    /-
      case intro.h
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (HMul.hMul (vi i) ((B (v i)) (v i))) 0
      ⊢ Eq (vi i) 0
    -/
  · exact eq_zero_of_ne_zero_of_mul_right_eq_zero (ho i) hB
    /-
      🎉 no goals
    -/
    /-
      case intro.h.h₀
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
      ⊢ ∀ (b : n), Membership.mem vi.support b → Ne b i → Eq (HMul.hMul (vi b) ((B ( …
    -/
  · intro j _ hij
    /-
      case intro.h.h₀
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
      j : n
      a✝ : Membership.mem vi.support j
      hij : Ne j i
      ⊢ Eq (HMul.hMul (vi j) ((B (v j)) (v i))) 0
    -/
    convert mul_zero (vi j) using 2
    /-
      case h.e'_2.h.e'_6
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
      j : n
      a✝ : Membership.mem vi.support j
      hij : Ne j i
      ⊢ Eq ((B (v j)) (v i)) 0
    -/
    exact hO hij
    /-
      🎉 no goals
    -/
    /-
      case intro.h.h₁
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
      ⊢ Not (Membership.mem vi.support i) → Eq (HMul.hMul (vi i) ((B (v i)) (v i))) 0
    -/
  · intro hi
    /-
      case intro.h.h₁
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
      hi : Not (Membership.mem vi.support i)
      ⊢ Eq (HMul.hMul (vi i) ((B (v i)) (v i))) 0
    -/
    convert zero_mul (M₀ := R) _ using 2
    /-
      case h.e'_2.h.e'_5
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      n : Type w
      inst✝¹ : Nontrivial R
      inst✝ : NoZeroDivisors R
      B : LinearMap.BilinForm R M
      v : Basis n R M
      hO : B.iIsOrtho ⇑v
      ho : ∀ (i : n), Not (B.IsOrtho (v i) (v i))
      vi : Finsupp n R
      i : n
      hB : Eq (vi.support.sum fun x => HMul.hMul (vi x) ((B (v x)) (v i))) 0
      hi : Not (Membership.mem vi.support i)
      ⊢ Eq (vi i) 0
    -/
    exact Finsupp.not_mem_support_iff.mp hi
    /-
      🎉 no goals
    -/


theorem toLin_restrict_ker_eq_inf_orthogonal (B : BilinForm K V) (W : Subspace K V) (b : B.IsRefl) :
    (B.domRestrict W).ker.map W.subtype = (W ⊓ B.orthogonal ⊤ : Subspace K V) := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    W : Subspace K V
    b : B.IsRefl
    ⊢ Eq (Submodule.map (Submodule.subtype W) (LinearMap.ker (LinearMap.domRestric …
  -/
  ext x; constructor <;> intro hx
    /-
      case h.mp
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Submodule.map (Submodule.subtype W) (LinearMap.ker (Linea …
      ⊢ Membership.mem (Min.min W (B.orthogonal Top.top)) x
    -/
  · rcases hx with ⟨⟨x, hx⟩, hker, rfl⟩
    /-
      case h.mp.intro.mk.intro
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem W x
      hker : Membership.mem ↑(LinearMap.ker (LinearMap.domRestrict B W)) ⟨x, hx⟩
      ⊢ Membership.mem (Min.min W (B.orthogonal Top.top)) ((Submodule.subtype W) ⟨x, …
    -/
    erw [LinearMap.mem_ker] at hker
    /-
      case h.mp.intro.mk.intro
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem W x
      hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
      ⊢ Membership.mem (Min.min W (B.orthogonal Top.top)) ((Submodule.subtype W) ⟨x, …
    -/
    constructor
      /-
        case h.mp.intro.mk.intro.left
        V : Type u_5
        K : Type u_6
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        B : LinearMap.BilinForm K V
        W : Subspace K V
        b : B.IsRefl
        x : V
        hx : Membership.mem W x
        hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
        ⊢ Membership.mem (↑W) ((Submodule.subtype W) ⟨x, hx⟩)
      -/
    · simp [hx]
      /-
        🎉 no goals
      -/
      /-
        case h.mp.intro.mk.intro.right
        V : Type u_5
        K : Type u_6
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        B : LinearMap.BilinForm K V
        W : Subspace K V
        b : B.IsRefl
        x : V
        hx : Membership.mem W x
        hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
        ⊢ Membership.mem (↑(B.orthogonal Top.top)) ((Submodule.subtype W) ⟨x, hx⟩)
      -/
    · intro y _
      /-
        case h.mp.intro.mk.intro.right
        V : Type u_5
        K : Type u_6
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        B : LinearMap.BilinForm K V
        W : Subspace K V
        b : B.IsRefl
        x : V
        hx : Membership.mem W x
        hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
        y : V
        a✝ : Membership.mem Top.top y
        ⊢ B.IsOrtho y ((Submodule.subtype W) ⟨x, hx⟩)
      -/
      rw [IsOrtho, b]
      /-
        case h.mp.intro.mk.intro.right.a
        V : Type u_5
        K : Type u_6
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        B : LinearMap.BilinForm K V
        W : Subspace K V
        b : B.IsRefl
        x : V
        hx : Membership.mem W x
        hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
        y : V
        a✝ : Membership.mem Top.top y
        ⊢ Eq ((B ((Submodule.subtype W) ⟨x, hx⟩)) y) 0
      -/
      change (B.domRestrict W) ⟨x, hx⟩ y = 0
      /-
        case h.mp.intro.mk.intro.right.a
        V : Type u_5
        K : Type u_6
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        B : LinearMap.BilinForm K V
        W : Subspace K V
        b : B.IsRefl
        x : V
        hx : Membership.mem W x
        hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
        y : V
        a✝ : Membership.mem Top.top y
        ⊢ Eq (((LinearMap.domRestrict B W) ⟨x, hx⟩) y) 0
      -/
      rw [hker]
      /-
        case h.mp.intro.mk.intro.right.a
        V : Type u_5
        K : Type u_6
        inst✝² : Field K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        B : LinearMap.BilinForm K V
        W : Subspace K V
        b : B.IsRefl
        x : V
        hx : Membership.mem W x
        hker : Eq ((LinearMap.domRestrict B W) ⟨x, hx⟩) 0
        y : V
        a✝ : Membership.mem Top.top y
        ⊢ Eq (0 y) 0
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Min.min W (B.orthogonal Top.top)) x
      ⊢ Membership.mem (Submodule.map (Submodule.subtype W) (LinearMap.ker (LinearMa …
    -/
  · simp_rw [Submodule.mem_map, LinearMap.mem_ker]
    /-
      case h.mpr
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Min.min W (B.orthogonal Top.top)) x
      ⊢ Exists fun y => And (Eq ((LinearMap.domRestrict B W) y) 0) (Eq ((Submodule.s …
    -/
    refine ⟨⟨x, hx.1⟩, ?_, rfl⟩
    /-
      case h.mpr
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Min.min W (B.orthogonal Top.top)) x
      ⊢ Eq ((LinearMap.domRestrict B W) ⟨x, ⋯⟩) 0
    -/
    ext y
    /-
      case h.mpr.h
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Min.min W (B.orthogonal Top.top)) x
      y : V
      ⊢ Eq (((LinearMap.domRestrict B W) ⟨x, ⋯⟩) y) (0 y)
    -/
    change B x y = 0
    /-
      case h.mpr.h
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Min.min W (B.orthogonal Top.top)) x
      y : V
      ⊢ Eq ((B x) y) 0
    -/
    rw [b]
    /-
      case h.mpr.h.a
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      b : B.IsRefl
      x : V
      hx : Membership.mem (Min.min W (B.orthogonal Top.top)) x
      y : V
      ⊢ Eq ((B y) x) 0
    -/
    exact hx.2 _ Submodule.mem_top
    /-
      🎉 no goals
    -/


theorem toLin_restrict_range_dualCoannihilator_eq_orthogonal (B : BilinForm K V)
    (W : Subspace K V) : (B.domRestrict W).range.dualCoannihilator = B.orthogonal W := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    W : Subspace K V
    ⊢ Eq (LinearMap.range (LinearMap.domRestrict B W)).dualCoannihilator (B.orthog …
  -/
  ext x; constructor <;> rw [mem_orthogonal_iff] <;> intro hx
    /-
      case h.mp
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      x : V
      hx : Membership.mem (LinearMap.range (LinearMap.domRestrict B W)).dualCoannihi …
      ⊢ ∀ (n : V), Membership.mem W n → B.IsOrtho n x
    -/
  · intro y hy
    /-
      case h.mp
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      x : V
      hx : Membership.mem (LinearMap.range (LinearMap.domRestrict B W)).dualCoannihi …
      y : V
      hy : Membership.mem W y
      ⊢ B.IsOrtho y x
    -/
    rw [Submodule.mem_dualCoannihilator] at hx
    /-
      case h.mp
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      x : V
      hx : ∀ (φ : Module.Dual K V), Membership.mem (LinearMap.range (LinearMap.domRe …
      y : V
      hy : Membership.mem W y
      ⊢ B.IsOrtho y x
    -/
    exact hx (B.domRestrict W ⟨y, hy⟩) ⟨⟨y, hy⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      x : V
      hx : ∀ (n : V), Membership.mem W n → B.IsOrtho n x
      ⊢ Membership.mem (LinearMap.range (LinearMap.domRestrict B W)).dualCoannihilat …
    -/
  · rw [Submodule.mem_dualCoannihilator]
    /-
      case h.mpr
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      x : V
      hx : ∀ (n : V), Membership.mem W n → B.IsOrtho n x
      ⊢ ∀ (φ : Module.Dual K V), Membership.mem (LinearMap.range (LinearMap.domRestr …
    -/
    rintro _ ⟨⟨w, hw⟩, rfl⟩
    /-
      case h.mpr.intro.mk
      V : Type u_5
      K : Type u_6
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      B : LinearMap.BilinForm K V
      W : Subspace K V
      x : V
      hx : ∀ (n : V), Membership.mem W n → B.IsOrtho n x
      w : V
      hw : Membership.mem W w
      ⊢ Eq (((LinearMap.domRestrict B W) ⟨w, hw⟩) x) 0
    -/
    exact hx w hw
    /-
      🎉 no goals
    -/


lemma ker_restrict_eq_of_codisjoint {p q : Submodule R M} (hpq : Codisjoint p q)
    {B : LinearMap.BilinForm R M} (hB : ∀ x ∈ p, ∀ y ∈ q, B x y = 0) :
    LinearMap.ker (B.restrict p) = (LinearMap.ker B).comap p.subtype := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p q : Submodule R M
    hpq : Codisjoint p q
    B : LinearMap.BilinForm R M
    hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
    ⊢ Eq (LinearMap.ker (B.restrict p)) (Submodule.comap p.subtype (LinearMap.ker  …
  -/
  ext ⟨z, hz⟩
  /-
    case h.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p q : Submodule R M
    hpq : Codisjoint p q
    B : LinearMap.BilinForm R M
    hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
    z : M
    hz : Membership.mem p z
    ⊢ Iff (Membership.mem (LinearMap.ker (B.restrict p)) ⟨z, hz⟩) (Membership.mem  …
  -/
  simp only [LinearMap.mem_ker, Submodule.mem_comap, Submodule.coe_subtype]
  /-
    case h.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p q : Submodule R M
    hpq : Codisjoint p q
    B : LinearMap.BilinForm R M
    hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
    z : M
    hz : Membership.mem p z
    ⊢ Iff (Eq ((B.restrict p) ⟨z, hz⟩) 0) (Eq (B z) 0)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case h.mk.refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      p q : Submodule R M
      hpq : Codisjoint p q
      B : LinearMap.BilinForm R M
      hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
      z : M
      hz : Membership.mem p z
      h : Eq ((B.restrict p) ⟨z, hz⟩) 0
      ⊢ Eq (B z) 0
    -/
  · ext w
    /-
      case h.mk.refine_1.h
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      p q : Submodule R M
      hpq : Codisjoint p q
      B : LinearMap.BilinForm R M
      hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
      z : M
      hz : Membership.mem p z
      h : Eq ((B.restrict p) ⟨z, hz⟩) 0
      w : M
      ⊢ Eq ((B z) w) (0 w)
    -/
    obtain ⟨x, hx, y, hy, rfl⟩ := Submodule.exists_add_eq_of_codisjoint hpq w
    /-
      case h.mk.refine_1.h.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      p q : Submodule R M
      hpq : Codisjoint p q
      B : LinearMap.BilinForm R M
      hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
      z : M
      hz : Membership.mem p z
      h : Eq ((B.restrict p) ⟨z, hz⟩) 0
      x : M
      hx : Membership.mem p x
      y : M
      hy : Membership.mem q y
      ⊢ Eq ((B z) (HAdd.hAdd x y)) (0 (HAdd.hAdd x y))
    -/
    simpa [hB z hz y hy] using LinearMap.congr_fun h ⟨x, hx⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      p q : Submodule R M
      hpq : Codisjoint p q
      B : LinearMap.BilinForm R M
      hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
      z : M
      hz : Membership.mem p z
      h : Eq (B z) 0
      ⊢ Eq ((B.restrict p) ⟨z, hz⟩) 0
    -/
  · ext ⟨x, hx⟩
    /-
      case h.mk.refine_2.h.mk
      R : Type u_1
      M : Type u_2
      inst✝² : CommSemiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      p q : Submodule R M
      hpq : Codisjoint p q
      B : LinearMap.BilinForm R M
      hB : ∀ (x : M), Membership.mem p x → ∀ (y : M), Membership.mem q y → Eq ((B x) …
      z : M
      hz : Membership.mem p z
      h : Eq (B z) 0
      x : M
      hx : Membership.mem p x
      ⊢ Eq (((B.restrict p) ⟨z, hz⟩) ⟨x, hx⟩) (0 ⟨x, hx⟩)
    -/
    simpa using LinearMap.congr_fun h x
    /-
      🎉 no goals
    -/


lemma inf_orthogonal_self_le_ker_restrict {W : Submodule R M} (b₁ : B.IsRefl) :
    W ⊓ B.orthogonal W ≤ (LinearMap.ker <| B.restrict W).map W.subtype := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    W : Submodule R M
    b₁ : B.IsRefl
    ⊢ LE.le (Min.min W (B.orthogonal W)) (Submodule.map W.subtype (LinearMap.ker ( …
  -/
  rintro v ⟨hv : v ∈ W, hv' : v ∈ B.orthogonal W⟩
  simp only [Submodule.mem_map, mem_ker, restrict_apply, Submodule.coe_subtype, Subtype.exists,
    exists_and_left, exists_prop, exists_eq_right_right]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    W : Submodule R M
    b₁ : B.IsRefl
    v : M
    hv : Membership.mem W v
    hv' : Membership.mem (B.orthogonal W) v
    ⊢ And (Eq ((B v).domRestrict W) 0) (Membership.mem W v)
  -/
  refine ⟨?_, hv⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    W : Submodule R M
    b₁ : B.IsRefl
    v : M
    hv : Membership.mem W v
    hv' : Membership.mem (B.orthogonal W) v
    ⊢ Eq ((B v).domRestrict W) 0
  -/
  ext ⟨w, hw⟩
  /-
    case intro.h.mk
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    B : LinearMap.BilinForm R M
    W : Submodule R M
    b₁ : B.IsRefl
    v : M
    hv : Membership.mem W v
    hv' : Membership.mem (B.orthogonal W) v
    w : M
    hw : Membership.mem W w
    ⊢ Eq (((B v).domRestrict W) ⟨w, hw⟩) (0 ⟨w, hw⟩)
  -/
  exact b₁ w v <| hv' w hw
  /-
    🎉 no goals
  -/


theorem finrank_add_finrank_orthogonal (b₁ : B.IsRefl) (W : Submodule K V) :
    finrank K W + finrank K (B.orthogonal W) =
      finrank K V + finrank K (W ⊓ B.orthogonal ⊤ : Subspace K V) := by
  rw [← toLin_restrict_ker_eq_inf_orthogonal _ _ b₁, ←
    toLin_restrict_range_dualCoannihilator_eq_orthogonal _ _, finrank_map_subtype_eq]
  conv_rhs =>
    rw [← @Subspace.finrank_add_finrank_dualCoannihilator_eq K V _ _ _ _
        (LinearMap.range (B.domRestrict W)),
      add_comm, ← add_assoc, add_comm (finrank K (LinearMap.ker (B.domRestrict W))),
      LinearMap.finrank_range_add_finrank_ker]


lemma finrank_orthogonal (hB : B.Nondegenerate) (hB₀ : B.IsRefl) (W : Submodule K V) :
    finrank K (B.orthogonal W) = finrank K V - finrank K W := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB₀ : B.IsRefl
    W : Submodule K V
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (B.orthogonal W) x)) ( …
  -/
  have := finrank_add_finrank_orthogonal hB₀ (W := W)
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB₀ : B.IsRefl
    W : Submodule K V
    this : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem W x))  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (B.orthogonal W) x)) ( …
  -/
  rw [B.orthogonal_top_eq_bot hB hB₀, inf_bot_eq, finrank_bot, add_zero] at this
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB₀ : B.IsRefl
    W : Submodule K V
    this : Eq (HAdd.hAdd (Module.finrank K (Subtype fun x => Membership.mem W x))  …
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (B.orthogonal W) x)) ( …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma orthogonal_orthogonal (hB : B.Nondegenerate) (hB₀ : B.IsRefl) (W : Submodule K V) :
    B.orthogonal (B.orthogonal W) = W := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB₀ : B.IsRefl
    W : Submodule K V
    ⊢ Eq (B.orthogonal (B.orthogonal W)) W
  -/
  apply (eq_of_le_of_finrank_le (LinearMap.BilinForm.le_orthogonal_orthogonal hB₀) _).symm
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB₀ : B.IsRefl
    W : Submodule K V
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (B.orthogonal (B.or …
  -/
  simp only [finrank_orthogonal hB hB₀]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    hB : B.Nondegenerate
    hB₀ : B.IsRefl
    W : Submodule K V
    ⊢ LE.le (HSub.hSub (Module.finrank K V) (HSub.hSub (Module.finrank K V) (Modul …
  -/
  omega
  /-
    🎉 no goals
  -/


lemma isCompl_orthogonal_iff_disjoint (hB₀ : B.IsRefl) :
    IsCompl W (B.orthogonal W) ↔ Disjoint W (B.orthogonal W) := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    hB₀ : B.IsRefl
    ⊢ Iff (IsCompl W (B.orthogonal W)) (Disjoint W (B.orthogonal W))
  -/
  refine ⟨IsCompl.disjoint, fun h ↦ ⟨h, ?_⟩⟩
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    hB₀ : B.IsRefl
    h : Disjoint W (B.orthogonal W)
    ⊢ Codisjoint W (B.orthogonal W)
  -/
  rw [codisjoint_iff]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    hB₀ : B.IsRefl
    h : Disjoint W (B.orthogonal W)
    ⊢ Eq (Max.max W (B.orthogonal W)) Top.top
  -/
  apply (eq_top_of_finrank_eq <| (finrank_le _).antisymm _)
  calc
    finrank K V ≤ finrank K V + finrank K ↥(W ⊓ B.orthogonal ⊤) := le_self_add
    _ ≤ finrank K ↥(W ⊔ B.orthogonal W) + finrank K ↥(W ⊓ B.orthogonal W) := ?_
    _ ≤ finrank K ↥(W ⊔ B.orthogonal W) := by simp [h.eq_bot]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    hB₀ : B.IsRefl
    h : Disjoint W (B.orthogonal W)
    ⊢ LE.le (HAdd.hAdd (Module.finrank K V) (Module.finrank K (Subtype fun x => Me …
  -/
  rw [finrank_sup_add_finrank_inf_eq, finrank_add_finrank_orthogonal hB₀ W]
  /-
    🎉 no goals
  -/


/-- A subspace is complement to its orthogonal complement with respect to some
reflexive bilinear form if that bilinear form restricted on to the subspace is nondegenerate. -/
theorem isCompl_orthogonal_of_restrict_nondegenerate
    (b₁ : B.IsRefl) (b₂ : (B.restrict W).Nondegenerate) : IsCompl W (B.orthogonal W) := by
  have : W ⊓ B.orthogonal W = ⊥ := by
    rw [eq_bot_iff]
    intro x hx
    obtain ⟨hx₁, hx₂⟩ := mem_inf.1 hx
    refine Subtype.mk_eq_mk.1 (b₂ ⟨x, hx₁⟩ ?_)
    rintro ⟨n, hn⟩
    simp only [restrict_apply, domRestrict_apply]
    exact b₁ n x (b₁ x n (b₁ n x (hx₂ n hn)))
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    this : Eq (Min.min W (B.orthogonal W)) Bot.bot
    ⊢ IsCompl W (B.orthogonal W)
  -/
  refine IsCompl.of_eq this (eq_top_of_finrank_eq <| (finrank_le _).antisymm ?_)
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    this : Eq (Min.min W (B.orthogonal W)) Bot.bot
    ⊢ LE.le (Module.finrank K V) (Module.finrank K (Subtype fun x => Membership.me …
  -/
  conv_rhs => rw [← add_zero (finrank K _)]
  rw [← finrank_bot K V, ← this, finrank_sup_add_finrank_inf_eq,
    finrank_add_finrank_orthogonal b₁]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    this : Eq (Min.min W (B.orthogonal W)) Bot.bot
    ⊢ LE.le (Module.finrank K V) (HAdd.hAdd (Module.finrank K V) (Module.finrank K …
  -/
  exact le_self_add
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-24")]
alias restrict_nondegenerate_of_isCompl_orthogonal := isCompl_orthogonal_of_restrict_nondegenerate


/-- A subspace is complement to its orthogonal complement with respect to some reflexive bilinear
form if and only if that bilinear form restricted on to the subspace is nondegenerate. -/
theorem restrict_nondegenerate_iff_isCompl_orthogonal
    (b₁ : B.IsRefl) : (B.restrict W).Nondegenerate ↔ IsCompl W (B.orthogonal W) :=
  ⟨fun b₂ => isCompl_orthogonal_of_restrict_nondegenerate b₁ b₂, fun h =>
    B.nondegenerate_restrict_of_disjoint_orthogonal b₁ h.1⟩


lemma orthogonal_eq_top_iff (b₁ : B.IsRefl) (b₂ : (B.restrict W).Nondegenerate) :
    B.orthogonal W = ⊤ ↔ W = ⊥ := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    ⊢ Iff (Eq (B.orthogonal W) Top.top) (Eq W Bot.bot)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by simp [h]⟩
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    h : Eq (B.orthogonal W) Top.top
    ⊢ Eq W Bot.bot
  -/
  have := (B.isCompl_orthogonal_of_restrict_nondegenerate b₁ b₂).inf_eq_bot
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    h : Eq (B.orthogonal W) Top.top
    this : Eq (Min.min W (B.orthogonal W)) Bot.bot
    ⊢ Eq W Bot.bot
  -/
  rwa [h, inf_top_eq] at this
  /-
    🎉 no goals
  -/


lemma eq_top_of_restrict_nondegenerate_of_orthogonal_eq_bot
    (b₁ : B.IsRefl) (b₂ : (B.restrict W).Nondegenerate) (b₃ : B.orthogonal W = ⊥) :
    W = ⊤ := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    b₃ : Eq (B.orthogonal W) Bot.bot
    ⊢ Eq W Top.top
  -/
  have := (B.isCompl_orthogonal_of_restrict_nondegenerate b₁ b₂).sup_eq_top
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    b₃ : Eq (B.orthogonal W) Bot.bot
    this : Eq (Max.max W (B.orthogonal W)) Top.top
    ⊢ Eq W Top.top
  -/
  rwa [b₃, sup_bot_eq] at this
  /-
    🎉 no goals
  -/


lemma orthogonal_eq_bot_iff
    (b₁ : B.IsRefl) (b₂ : (B.restrict W).Nondegenerate) (b₃ : B.Nondegenerate) :
    B.orthogonal W = ⊥ ↔ W = ⊤ := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    b₃ : B.Nondegenerate
    ⊢ Iff (Eq (B.orthogonal W) Bot.bot) (Eq W Top.top)
  -/
  refine ⟨eq_top_of_restrict_nondegenerate_of_orthogonal_eq_bot b₁ b₂, fun h ↦ ?_⟩
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    b₃ : B.Nondegenerate
    h : Eq W Top.top
    ⊢ Eq (B.orthogonal W) Bot.bot
  -/
  rw [h, eq_bot_iff]
  /-
    V : Type u_5
    K : Type u_6
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    B : LinearMap.BilinForm K V
    W : Submodule K V
    b₁ : B.IsRefl
    b₂ : (B.restrict W).Nondegenerate
    b₃ : B.Nondegenerate
    h : Eq W Top.top
    ⊢ LE.le (B.orthogonal Top.top) Bot.bot
  -/
  exact fun x hx ↦ b₃ x fun y ↦ b₁ y x <| by simpa using hx y
  /-
    🎉 no goals
  -/


/-- The restriction of a reflexive, non-degenerate bilinear form on the orthogonal complement of
the span of a singleton is also non-degenerate. -/
theorem restrict_nondegenerate_orthogonal_spanSingleton (B : BilinForm K V) (b₁ : B.Nondegenerate)
    (b₂ : B.IsRefl) {x : V} (hx : ¬B.IsOrtho x x) :
    Nondegenerate <| B.restrict <| B.orthogonal (K ∙ x) := by
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    b₁ : B.Nondegenerate
    b₂ : B.IsRefl
    x : V
    hx : Not (B.IsOrtho x x)
    ⊢ (B.restrict (B.orthogonal (Submodule.span K (Singleton.singleton x)))).Nonde …
  -/
  refine fun m hm => Submodule.coe_eq_zero.1 (b₁ m.1 fun n => ?_)
  have : n ∈ (K ∙ x) ⊔ B.orthogonal (K ∙ x) :=
    (span_singleton_sup_orthogonal_eq_top hx).symm ▸ Submodule.mem_top
  /-
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    b₁ : B.Nondegenerate
    b₂ : B.IsRefl
    x : V
    hx : Not (B.IsOrtho x x)
    m : Subtype fun x_1 => Membership.mem (B.orthogonal (Submodule.span K (Singlet …
    hm : ∀ (n : Subtype fun x_1 => Membership.mem (B.orthogonal (Submodule.span K  …
    n : V
    this : Membership.mem (Max.max (Submodule.span K (Singleton.singleton x)) (B.o …
    ⊢ Eq ((B ↑m) n) 0
  -/
  rcases Submodule.mem_sup.1 this with ⟨y, hy, z, hz, rfl⟩
  /-
    case intro.intro.intro.intro
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    b₁ : B.Nondegenerate
    b₂ : B.IsRefl
    x : V
    hx : Not (B.IsOrtho x x)
    m : Subtype fun x_1 => Membership.mem (B.orthogonal (Submodule.span K (Singlet …
    hm : ∀ (n : Subtype fun x_1 => Membership.mem (B.orthogonal (Submodule.span K  …
    y : V
    hy : Membership.mem (Submodule.span K (Singleton.singleton x)) y
    z : V
    hz : Membership.mem (B.orthogonal (Submodule.span K (Singleton.singleton x))) z
    this : Membership.mem (Max.max (Submodule.span K (Singleton.singleton x)) (B.o …
    ⊢ Eq ((B ↑m) (HAdd.hAdd y z)) 0
  -/
  specialize hm ⟨z, hz⟩
  /-
    case intro.intro.intro.intro
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    b₁ : B.Nondegenerate
    b₂ : B.IsRefl
    x : V
    hx : Not (B.IsOrtho x x)
    m : Subtype fun x_1 => Membership.mem (B.orthogonal (Submodule.span K (Singlet …
    y : V
    hy : Membership.mem (Submodule.span K (Singleton.singleton x)) y
    z : V
    hz : Membership.mem (B.orthogonal (Submodule.span K (Singleton.singleton x))) z
    this : Membership.mem (Max.max (Submodule.span K (Singleton.singleton x)) (B.o …
    hm : Eq (((B.restrict (B.orthogonal (Submodule.span K (Singleton.singleton x)) …
    ⊢ Eq ((B ↑m) (HAdd.hAdd y z)) 0
  -/
  rw [restrict] at hm
  /-
    case intro.intro.intro.intro
    V : Type u_5
    K : Type u_6
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    B : LinearMap.BilinForm K V
    b₁ : B.Nondegenerate
    b₂ : B.IsRefl
    x : V
    hx : Not (B.IsOrtho x x)
    m : Subtype fun x_1 => Membership.mem (B.orthogonal (Submodule.span K (Singlet …
    y : V
    hy : Membership.mem (Submodule.span K (Singleton.singleton x)) y
    z : V
    hz : Membership.mem (B.orthogonal (Submodule.span K (Singleton.singleton x))) z
    this : Membership.mem (Max.max (Submodule.span K (Singleton.singleton x)) (B.o …
    hm : Eq (((LinearMap.domRestrict₁₂ B (B.orthogonal (Submodule.span K (Singleto …
    ⊢ Eq ((B ↑m) (HAdd.hAdd y z)) 0
  -/
  erw [add_right, show B m.1 y = 0 by rw [b₂]; exact m.2 y hy, hm, add_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-30")]
alias restrictNondegenerateOrthogonalSpanSingleton :=
  restrict_nondegenerate_orthogonal_spanSingleton


