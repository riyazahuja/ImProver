/-- The maps `f` and `g` form an exact pair :
  `g y = 0` iff `y` belongs to the image of `f` -/
def Exact [Zero P] : Prop := ∀ y, g y = 0 ↔ y ∈ Set.range f


lemma apply_apply_eq_zero [Zero P] (h : Exact f g) (x : M) :
    g (f x) = 0 := (h _).mpr <| Set.mem_range_self _


lemma comp_eq_zero [Zero P] (h : Exact f g) : g.comp f = 0 :=
  funext h.apply_apply_eq_zero


lemma of_comp_of_mem_range [Zero P] (h1 : g ∘ f = 0)
    (h2 : ∀ x, g x = 0 → x ∈ Set.range f) : Exact f g :=
  fun y => Iff.intro (h2 y) <|
    Exists.rec ((forall_apply_eq_imp_iff (p := (g · = 0))).mpr (congrFun h1) y)


lemma comp_injective [Zero P] [Zero P'] (exact : Exact f g)
    (inj : Function.Injective g') (h0 : g' 0 = 0) :
    Exact f (g' ∘ g) := by
  /-
    M : Type u_2
    N : Type u_4
    P : Type u_6
    P' : Type u_7
    f : M → N
    g : N → P
    g' : P → P'
    inst✝¹ : Zero P
    inst✝ : Zero P'
    exact : Function.Exact f g
    inj : Function.Injective g'
    h0 : Eq (g' 0) 0
    ⊢ Function.Exact f (Function.comp g' g)
  -/
  intro x
  /-
    M : Type u_2
    N : Type u_4
    P : Type u_6
    P' : Type u_7
    f : M → N
    g : N → P
    g' : P → P'
    inst✝¹ : Zero P
    inst✝ : Zero P'
    exact : Function.Exact f g
    inj : Function.Injective g'
    h0 : Eq (g' 0) 0
    x : N
    ⊢ Iff (Eq (Function.comp g' g x) 0) (Membership.mem (Set.range f) x)
  -/
  refine ⟨fun H => exact x |>.mp <| inj <| h0 ▸ H, ?_⟩
  /-
    M : Type u_2
    N : Type u_4
    P : Type u_6
    P' : Type u_7
    f : M → N
    g : N → P
    g' : P → P'
    inst✝¹ : Zero P
    inst✝ : Zero P'
    exact : Function.Exact f g
    inj : Function.Injective g'
    h0 : Eq (g' 0) 0
    x : N
    ⊢ Membership.mem (Set.range f) x → Eq (Function.comp g' g x) 0
  -/
  intro H
  /-
    M : Type u_2
    N : Type u_4
    P : Type u_6
    P' : Type u_7
    f : M → N
    g : N → P
    g' : P → P'
    inst✝¹ : Zero P
    inst✝ : Zero P'
    exact : Function.Exact f g
    inj : Function.Injective g'
    h0 : Eq (g' 0) 0
    x : N
    H : Membership.mem (Set.range f) x
    ⊢ Eq (Function.comp g' g x) 0
  -/
  rw [Function.comp_apply, exact x |>.mpr H, h0]
  /-
    🎉 no goals
  -/


lemma of_comp_eq_zero_of_ker_in_range [Zero P] (hc : g.comp f = 0)
    (hr : ∀ y, g y = 0 → y ∈ Set.range f) :
    Exact f g :=
  fun y ↦ ⟨hr y, fun ⟨x, hx⟩ ↦ hx ▸ congrFun hc x⟩


lemma exact_iff :
    Exact f g ↔ ker g = range f :=
  Iff.symm SetLike.ext_iff


lemma exact_of_comp_eq_zero_of_ker_le_range
    (h1 : g.comp f = 0) (h2 : ker g ≤ range f) : Exact f g :=
  Exact.of_comp_of_mem_range (congrArg DFunLike.coe h1) h2


lemma exact_of_comp_of_mem_range
    (h1 : g.comp f = 0) (h2 : ∀ x, g x = 0 → x ∈ range f) : Exact f g :=
  exact_of_comp_eq_zero_of_ker_le_range h1 h2


/-- When we have a commutative diagram from a sequence of two maps to another,
such that the left vertical map is surjective, the middle vertical map is bijective and the right
vertical map is injective, then the upper row is exact iff the lower row is.
See `ShortComplex.exact_iff_of_epi_of_isIso_of_mono` in the file
`Algebra.Homology.ShortComplex.Exact` for the categorical version of this result. -/
lemma exact_iff_of_surjective_of_bijective_of_injective
  {M₁ M₂ M₃ N₁ N₂ N₃ : Type*} [AddCommMonoid M₁] [AddCommMonoid M₂] [AddCommMonoid M₃]
  [AddCommMonoid N₁] [AddCommMonoid N₂] [AddCommMonoid N₃]
  (f : M₁ →+ M₂) (g : M₂ →+ M₃) (f' : N₁ →+ N₂) (g' : N₂ →+ N₃)
  (τ₁ : M₁ →+ N₁) (τ₂ : M₂ →+ N₂) (τ₃ : M₃ →+ N₃)
  (comm₁₂ : f'.comp τ₁ = τ₂.comp f)
  (comm₂₃ : g'.comp τ₂ = τ₃.comp g)
  (h₁ : Function.Surjective τ₁) (h₂ : Function.Bijective τ₂) (h₃ : Function.Injective τ₃) :
    Exact f g ↔ Exact f' g' := by
  /-
    M₁ : Type u_8
    M₂ : Type u_9
    M₃ : Type u_10
    N₁ : Type u_11
    N₂ : Type u_12
    N₃ : Type u_13
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : AddCommMonoid N₁
    inst✝¹ : AddCommMonoid N₂
    inst✝ : AddCommMonoid N₃
    f : AddMonoidHom M₁ M₂
    g : AddMonoidHom M₂ M₃
    f' : AddMonoidHom N₁ N₂
    g' : AddMonoidHom N₂ N₃
    τ₁ : AddMonoidHom M₁ N₁
    τ₂ : AddMonoidHom M₂ N₂
    τ₃ : AddMonoidHom M₃ N₃
    comm₁₂ : Eq (f'.comp τ₁) (τ₂.comp f)
    comm₂₃ : Eq (g'.comp τ₂) (τ₃.comp g)
    h₁ : Function.Surjective ⇑τ₁
    h₂ : Function.Bijective ⇑τ₂
    h₃ : Function.Injective ⇑τ₃
    ⊢ Iff (Function.Exact ⇑f ⇑g) (Function.Exact ⇑f' ⇑g')
  -/
  replace comm₁₂ := DFunLike.congr_fun comm₁₂
  /-
    M₁ : Type u_8
    M₂ : Type u_9
    M₃ : Type u_10
    N₁ : Type u_11
    N₂ : Type u_12
    N₃ : Type u_13
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : AddCommMonoid N₁
    inst✝¹ : AddCommMonoid N₂
    inst✝ : AddCommMonoid N₃
    f : AddMonoidHom M₁ M₂
    g : AddMonoidHom M₂ M₃
    f' : AddMonoidHom N₁ N₂
    g' : AddMonoidHom N₂ N₃
    τ₁ : AddMonoidHom M₁ N₁
    τ₂ : AddMonoidHom M₂ N₂
    τ₃ : AddMonoidHom M₃ N₃
    comm₂₃ : Eq (g'.comp τ₂) (τ₃.comp g)
    h₁ : Function.Surjective ⇑τ₁
    h₂ : Function.Bijective ⇑τ₂
    h₃ : Function.Injective ⇑τ₃
    comm₁₂ : ∀ (x : M₁), Eq ((f'.comp τ₁) x) ((τ₂.comp f) x)
    ⊢ Iff (Function.Exact ⇑f ⇑g) (Function.Exact ⇑f' ⇑g')
  -/
  replace comm₂₃ := DFunLike.congr_fun comm₂₃
  /-
    M₁ : Type u_8
    M₂ : Type u_9
    M₃ : Type u_10
    N₁ : Type u_11
    N₂ : Type u_12
    N₃ : Type u_13
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : AddCommMonoid N₁
    inst✝¹ : AddCommMonoid N₂
    inst✝ : AddCommMonoid N₃
    f : AddMonoidHom M₁ M₂
    g : AddMonoidHom M₂ M₃
    f' : AddMonoidHom N₁ N₂
    g' : AddMonoidHom N₂ N₃
    τ₁ : AddMonoidHom M₁ N₁
    τ₂ : AddMonoidHom M₂ N₂
    τ₃ : AddMonoidHom M₃ N₃
    h₁ : Function.Surjective ⇑τ₁
    h₂ : Function.Bijective ⇑τ₂
    h₃ : Function.Injective ⇑τ₃
    comm₁₂ : ∀ (x : M₁), Eq ((f'.comp τ₁) x) ((τ₂.comp f) x)
    comm₂₃ : ∀ (x : M₂), Eq ((g'.comp τ₂) x) ((τ₃.comp g) x)
    ⊢ Iff (Function.Exact ⇑f ⇑g) (Function.Exact ⇑f' ⇑g')
  -/
  dsimp at comm₁₂ comm₂₃
  /-
    M₁ : Type u_8
    M₂ : Type u_9
    M₃ : Type u_10
    N₁ : Type u_11
    N₂ : Type u_12
    N₃ : Type u_13
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : AddCommMonoid N₁
    inst✝¹ : AddCommMonoid N₂
    inst✝ : AddCommMonoid N₃
    f : AddMonoidHom M₁ M₂
    g : AddMonoidHom M₂ M₃
    f' : AddMonoidHom N₁ N₂
    g' : AddMonoidHom N₂ N₃
    τ₁ : AddMonoidHom M₁ N₁
    τ₂ : AddMonoidHom M₂ N₂
    τ₃ : AddMonoidHom M₃ N₃
    h₁ : Function.Surjective ⇑τ₁
    h₂ : Function.Bijective ⇑τ₂
    h₃ : Function.Injective ⇑τ₃
    comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
    comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
    ⊢ Iff (Function.Exact ⇑f ⇑g) (Function.Exact ⇑f' ⇑g')
  -/
  constructor
    /-
      case mp
      M₁ : Type u_8
      M₂ : Type u_9
      M₃ : Type u_10
      N₁ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : AddCommMonoid N₁
      inst✝¹ : AddCommMonoid N₂
      inst✝ : AddCommMonoid N₃
      f : AddMonoidHom M₁ M₂
      g : AddMonoidHom M₂ M₃
      f' : AddMonoidHom N₁ N₂
      g' : AddMonoidHom N₂ N₃
      τ₁ : AddMonoidHom M₁ N₁
      τ₂ : AddMonoidHom M₂ N₂
      τ₃ : AddMonoidHom M₃ N₃
      h₁ : Function.Surjective ⇑τ₁
      h₂ : Function.Bijective ⇑τ₂
      h₃ : Function.Injective ⇑τ₃
      comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
      comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
      ⊢ Function.Exact ⇑f ⇑g → Function.Exact ⇑f' ⇑g'
    -/
  · intro h y₂
    /-
      case mp
      M₁ : Type u_8
      M₂ : Type u_9
      M₃ : Type u_10
      N₁ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : AddCommMonoid N₁
      inst✝¹ : AddCommMonoid N₂
      inst✝ : AddCommMonoid N₃
      f : AddMonoidHom M₁ M₂
      g : AddMonoidHom M₂ M₃
      f' : AddMonoidHom N₁ N₂
      g' : AddMonoidHom N₂ N₃
      τ₁ : AddMonoidHom M₁ N₁
      τ₂ : AddMonoidHom M₂ N₂
      τ₃ : AddMonoidHom M₃ N₃
      h₁ : Function.Surjective ⇑τ₁
      h₂ : Function.Bijective ⇑τ₂
      h₃ : Function.Injective ⇑τ₃
      comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
      comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
      h : Function.Exact ⇑f ⇑g
      y₂ : N₂
      ⊢ Iff (Eq (g' y₂) 0) (Membership.mem (Set.range ⇑f') y₂)
    -/
    obtain ⟨x₂, rfl⟩ := h₂.2 y₂
    /-
      case mp.intro
      M₁ : Type u_8
      M₂ : Type u_9
      M₃ : Type u_10
      N₁ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : AddCommMonoid N₁
      inst✝¹ : AddCommMonoid N₂
      inst✝ : AddCommMonoid N₃
      f : AddMonoidHom M₁ M₂
      g : AddMonoidHom M₂ M₃
      f' : AddMonoidHom N₁ N₂
      g' : AddMonoidHom N₂ N₃
      τ₁ : AddMonoidHom M₁ N₁
      τ₂ : AddMonoidHom M₂ N₂
      τ₃ : AddMonoidHom M₃ N₃
      h₁ : Function.Surjective ⇑τ₁
      h₂ : Function.Bijective ⇑τ₂
      h₃ : Function.Injective ⇑τ₃
      comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
      comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
      h : Function.Exact ⇑f ⇑g
      x₂ : M₂
      ⊢ Iff (Eq (g' (τ₂ x₂)) 0) (Membership.mem (Set.range ⇑f') (τ₂ x₂))
    -/
    constructor
      /-
        case mp.intro.mp
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₂ : M₂
        ⊢ Eq (g' (τ₂ x₂)) 0 → Membership.mem (Set.range ⇑f') (τ₂ x₂)
      -/
    · intro hx₂
      /-
        case mp.intro.mp
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₂ : M₂
        hx₂ : Eq (g' (τ₂ x₂)) 0
        ⊢ Membership.mem (Set.range ⇑f') (τ₂ x₂)
      -/
      obtain ⟨x₁, rfl⟩ := (h x₂).1 (h₃ (by simpa only [map_zero, comm₂₃] using hx₂))
      /-
        case mp.intro.mp.intro
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₁ : M₁
        hx₂ : Eq (g' (τ₂ (f x₁))) 0
        ⊢ Membership.mem (Set.range ⇑f') (τ₂ (f x₁))
      -/
      exact ⟨τ₁ x₁, by simp only [comm₁₂]⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.mpr
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₂ : M₂
        ⊢ Membership.mem (Set.range ⇑f') (τ₂ x₂) → Eq (g' (τ₂ x₂)) 0
      -/
    · rintro ⟨y₁, hy₁⟩
      /-
        case mp.intro.mpr.intro
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₂ : M₂
        y₁ : N₁
        hy₁ : Eq (f' y₁) (τ₂ x₂)
        ⊢ Eq (g' (τ₂ x₂)) 0
      -/
      obtain ⟨x₁, rfl⟩ := h₁ y₁
      /-
        case mp.intro.mpr.intro.intro
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₂ : M₂
        x₁ : M₁
        hy₁ : Eq (f' (τ₁ x₁)) (τ₂ x₂)
        ⊢ Eq (g' (τ₂ x₂)) 0
      -/
      rw [comm₂₃, (h x₂).2 _, map_zero]
      /-
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f ⇑g
        x₂ : M₂
        x₁ : M₁
        hy₁ : Eq (f' (τ₁ x₁)) (τ₂ x₂)
        ⊢ Membership.mem (Set.range ⇑f) x₂
      -/
      exact ⟨x₁, h₂.1 (by simpa only [comm₁₂] using hy₁)⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      M₁ : Type u_8
      M₂ : Type u_9
      M₃ : Type u_10
      N₁ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : AddCommMonoid N₁
      inst✝¹ : AddCommMonoid N₂
      inst✝ : AddCommMonoid N₃
      f : AddMonoidHom M₁ M₂
      g : AddMonoidHom M₂ M₃
      f' : AddMonoidHom N₁ N₂
      g' : AddMonoidHom N₂ N₃
      τ₁ : AddMonoidHom M₁ N₁
      τ₂ : AddMonoidHom M₂ N₂
      τ₃ : AddMonoidHom M₃ N₃
      h₁ : Function.Surjective ⇑τ₁
      h₂ : Function.Bijective ⇑τ₂
      h₃ : Function.Injective ⇑τ₃
      comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
      comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
      ⊢ Function.Exact ⇑f' ⇑g' → Function.Exact ⇑f ⇑g
    -/
  · intro h x₂
    /-
      case mpr
      M₁ : Type u_8
      M₂ : Type u_9
      M₃ : Type u_10
      N₁ : Type u_11
      N₂ : Type u_12
      N₃ : Type u_13
      inst✝⁵ : AddCommMonoid M₁
      inst✝⁴ : AddCommMonoid M₂
      inst✝³ : AddCommMonoid M₃
      inst✝² : AddCommMonoid N₁
      inst✝¹ : AddCommMonoid N₂
      inst✝ : AddCommMonoid N₃
      f : AddMonoidHom M₁ M₂
      g : AddMonoidHom M₂ M₃
      f' : AddMonoidHom N₁ N₂
      g' : AddMonoidHom N₂ N₃
      τ₁ : AddMonoidHom M₁ N₁
      τ₂ : AddMonoidHom M₂ N₂
      τ₃ : AddMonoidHom M₃ N₃
      h₁ : Function.Surjective ⇑τ₁
      h₂ : Function.Bijective ⇑τ₂
      h₃ : Function.Injective ⇑τ₃
      comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
      comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
      h : Function.Exact ⇑f' ⇑g'
      x₂ : M₂
      ⊢ Iff (Eq (g x₂) 0) (Membership.mem (Set.range ⇑f) x₂)
    -/
    constructor
      /-
        case mpr.mp
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₂ : M₂
        ⊢ Eq (g x₂) 0 → Membership.mem (Set.range ⇑f) x₂
      -/
    · intro hx₂
      /-
        case mpr.mp
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₂ : M₂
        hx₂ : Eq (g x₂) 0
        ⊢ Membership.mem (Set.range ⇑f) x₂
      -/
      obtain ⟨y₁, hy₁⟩ := (h (τ₂ x₂)).1 (by simp only [comm₂₃, hx₂, map_zero])
      /-
        case mpr.mp.intro
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₂ : M₂
        hx₂ : Eq (g x₂) 0
        y₁ : N₁
        hy₁ : Eq (f' y₁) (τ₂ x₂)
        ⊢ Membership.mem (Set.range ⇑f) x₂
      -/
      obtain ⟨x₁, rfl⟩ := h₁ y₁
      /-
        case mpr.mp.intro.intro
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₂ : M₂
        hx₂ : Eq (g x₂) 0
        x₁ : M₁
        hy₁ : Eq (f' (τ₁ x₁)) (τ₂ x₂)
        ⊢ Membership.mem (Set.range ⇑f) x₂
      -/
      exact ⟨x₁, h₂.1 (by simpa only [comm₁₂] using hy₁)⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.mpr
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₂ : M₂
        ⊢ Membership.mem (Set.range ⇑f) x₂ → Eq (g x₂) 0
      -/
    · rintro ⟨x₁, rfl⟩
      /-
        case mpr.mpr.intro
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₁ : M₁
        ⊢ Eq (g (f x₁)) 0
      -/
      apply h₃
      /-
        case mpr.mpr.intro.a
        M₁ : Type u_8
        M₂ : Type u_9
        M₃ : Type u_10
        N₁ : Type u_11
        N₂ : Type u_12
        N₃ : Type u_13
        inst✝⁵ : AddCommMonoid M₁
        inst✝⁴ : AddCommMonoid M₂
        inst✝³ : AddCommMonoid M₃
        inst✝² : AddCommMonoid N₁
        inst✝¹ : AddCommMonoid N₂
        inst✝ : AddCommMonoid N₃
        f : AddMonoidHom M₁ M₂
        g : AddMonoidHom M₂ M₃
        f' : AddMonoidHom N₁ N₂
        g' : AddMonoidHom N₂ N₃
        τ₁ : AddMonoidHom M₁ N₁
        τ₂ : AddMonoidHom M₂ N₂
        τ₃ : AddMonoidHom M₃ N₃
        h₁ : Function.Surjective ⇑τ₁
        h₂ : Function.Bijective ⇑τ₂
        h₃ : Function.Injective ⇑τ₃
        comm₁₂ : ∀ (x : M₁), Eq (f' (τ₁ x)) (τ₂ (f x))
        comm₂₃ : ∀ (x : M₂), Eq (g' (τ₂ x)) (τ₃ (g x))
        h : Function.Exact ⇑f' ⇑g'
        x₁ : M₁
        ⊢ Eq (τ₃ (g (f x₁))) (τ₃ 0)
      -/
      simp only [← comm₁₂, ← comm₂₃, h.apply_apply_eq_zero (τ₁ x₁), map_zero]
      /-
        🎉 no goals
      -/


lemma addMonoidHom_ker_eq (hfg : Exact f g) :
    ker g = range f :=
  SetLike.ext hfg


lemma addMonoidHom_comp_eq_zero (h : Exact f g) : g.comp f = 0 :=
  DFunLike.coe_injective h.comp_eq_zero


lemma iff_of_ladder_addEquiv (comm₁₂ : g₁₂.comp e₁ = AddMonoidHom.comp e₂ f₁₂)
    (comm₂₃ : g₂₃.comp e₂ = AddMonoidHom.comp e₃ f₂₃) : Exact g₁₂ g₂₃ ↔ Exact f₁₂ f₂₃ :=
  (exact_iff_of_surjective_of_bijective_of_injective _ _ _ _ e₁ e₂ e₃ comm₁₂ comm₂₃
    e₁.surjective e₂.bijective e₃.injective).symm


lemma of_ladder_addEquiv_of_exact (comm₁₂ : g₁₂.comp e₁ = AddMonoidHom.comp e₂ f₁₂)
    (comm₂₃ : g₂₃.comp e₂ = AddMonoidHom.comp e₃ f₂₃) (H : Exact f₁₂ f₂₃) : Exact g₁₂ g₂₃ :=
  (iff_of_ladder_addEquiv _ _ _ comm₁₂ comm₂₃).2 H


lemma of_ladder_addEquiv_of_exact' (comm₁₂ : g₁₂.comp e₁ = AddMonoidHom.comp e₂ f₁₂)
    (comm₂₃ : g₂₃.comp e₂ = AddMonoidHom.comp e₃ f₂₃) (H : Exact g₁₂ g₂₃) : Exact f₁₂ f₂₃ :=
  (iff_of_ladder_addEquiv _ _ _ comm₁₂ comm₂₃).1 H


lemma exact_iff :
    Exact f g ↔ LinearMap.ker g = LinearMap.range f :=
  Iff.symm SetLike.ext_iff


lemma exact_of_comp_eq_zero_of_ker_le_range
    (h1 : g ∘ₗ f = 0) (h2 : ker g ≤ range f) : Exact f g :=
  Exact.of_comp_of_mem_range (congrArg DFunLike.coe h1) h2


lemma exact_of_comp_of_mem_range
    (h1 : g ∘ₗ f = 0) (h2 : ∀ x, g x = 0 → x ∈ range f) : Exact f g :=
  exact_of_comp_eq_zero_of_ker_le_range h1 h2


lemma exact_subtype_mkQ (Q : Submodule R N) :
    Exact (Submodule.subtype Q) (Submodule.mkQ Q) := by
  /-
    R : Type u_8
    N : Type u_10
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    Q : Submodule R N
    ⊢ Function.Exact ⇑Q.subtype ⇑Q.mkQ
  -/
  rw [exact_iff, Submodule.ker_mkQ, Submodule.range_subtype Q]
  /-
    🎉 no goals
  -/


lemma exact_map_mkQ_range (f : M →ₗ[R] N) :
    Exact f (Submodule.mkQ (range f)) :=
  exact_iff.mpr <| Submodule.ker_mkQ _


lemma exact_subtype_ker_map (g : N →ₗ[R] P) :
    Exact (Submodule.subtype (ker g)) g :=
  exact_iff.mpr <| (Submodule.range_subtype _).symm


variable (f g) in
lemma LinearEquiv.conj_exact_iff_exact (e : N ≃ₗ[R] N') :
    Function.Exact (e ∘ₗ f) (g ∘ₗ (e.symm : N' →ₗ[R] N)) ↔ Exact f g := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    N' : Type u_5
    P : Type u_6
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid N'
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R N'
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    e : LinearEquiv (RingHom.id R) N N'
    ⊢ Iff (Function.Exact ⇑((↑e).comp f) ⇑(g.comp ↑e.symm)) (Function.Exact ⇑f ⇑g)
  -/
  simp_rw [LinearMap.exact_iff, LinearMap.ker_comp, ← e.map_eq_comap, LinearMap.range_comp]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    N' : Type u_5
    P : Type u_6
    inst✝⁸ : Semiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid N
    inst✝⁵ : AddCommMonoid N'
    inst✝⁴ : AddCommMonoid P
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module R N'
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    e : LinearEquiv (RingHom.id R) N N'
    ⊢ Iff (Eq (Submodule.map (↑e) (LinearMap.ker g)) (Submodule.map (↑e) (LinearMa …
  -/
  exact (Submodule.map_injective_of_injective e.injective).eq_iff
  /-
    🎉 no goals
  -/


lemma Exact.linearMap_ker_eq (hfg : Exact f g) : ker g = range f :=
  SetLike.ext hfg


lemma Exact.linearMap_comp_eq_zero (h : Exact f g) : g.comp f = 0 :=
  DFunLike.coe_injective h.comp_eq_zero


lemma Surjective.comp_exact_iff_exact {p : M' →ₗ[R] M} (h : Surjective p) :
    Exact (f ∘ₗ p) g ↔ Exact f g :=
  iff_of_eq <| forall_congr fun x =>
    congrArg (g x = 0 ↔ x ∈ ·) (h.range_comp f)


lemma Injective.comp_exact_iff_exact {i : P →ₗ[R] P'} (h : Injective i) :
    Exact f (i ∘ₗ g) ↔ Exact f g :=
  forall_congr' fun _ => iff_congr (LinearMap.map_eq_zero_iff _ h) Iff.rfl


lemma Exact.iff_of_ladder_linearEquiv
    (h₁₂ : g₁₂ ∘ₗ e₁ = e₂ ∘ₗ f₁₂) (h₂₃ : g₂₃ ∘ₗ e₂ = e₃ ∘ₗ f₂₃) :
    Exact g₁₂ g₂₃ ↔ Exact f₁₂ f₂₃ :=
  iff_of_ladder_addEquiv e₁.toAddEquiv e₂.toAddEquiv e₃.toAddEquiv
    (f₁₂ := f₁₂) (f₂₃ := f₂₃) (g₁₂ := g₁₂) (g₂₃ := g₂₃)
    (congr_arg LinearMap.toAddMonoidHom h₁₂) (congr_arg LinearMap.toAddMonoidHom h₂₃)


lemma Exact.of_ladder_linearEquiv_of_exact
    (h₁₂ : g₁₂ ∘ₗ e₁ = e₂ ∘ₗ f₁₂) (h₂₃ : g₂₃ ∘ₗ e₂ = e₃ ∘ₗ f₂₃)
    (H : Exact f₁₂ f₂₃) : Exact g₁₂ g₂₃ := by
  /-
    R : Type u_1
    M : Type u_2
    M' : Type u_3
    N : Type u_4
    N' : Type u_5
    P : Type u_6
    P' : Type u_7
    inst✝¹² : Semiring R
    inst✝¹¹ : AddCommMonoid M
    inst✝¹⁰ : AddCommMonoid M'
    inst✝⁹ : AddCommMonoid N
    inst✝⁸ : AddCommMonoid N'
    inst✝⁷ : AddCommMonoid P
    inst✝⁶ : AddCommMonoid P'
    inst✝⁵ : Module R M
    inst✝⁴ : Module R M'
    inst✝³ : Module R N
    inst✝² : Module R N'
    inst✝¹ : Module R P
    inst✝ : Module R P'
    f₁₂ : LinearMap (RingHom.id R) M N
    f₂₃ : LinearMap (RingHom.id R) N P
    g₁₂ : LinearMap (RingHom.id R) M' N'
    g₂₃ : LinearMap (RingHom.id R) N' P'
    e₁ : LinearEquiv (RingHom.id R) M M'
    e₂ : LinearEquiv (RingHom.id R) N N'
    e₃ : LinearEquiv (RingHom.id R) P P'
    h₁₂ : Eq (g₁₂.comp ↑e₁) ((↑e₂).comp f₁₂)
    h₂₃ : Eq (g₂₃.comp ↑e₂) ((↑e₃).comp f₂₃)
    H : Function.Exact ⇑f₁₂ ⇑f₂₃
    ⊢ Function.Exact ⇑g₁₂ ⇑g₂₃
  -/
  rwa [iff_of_ladder_linearEquiv h₁₂ h₂₃]
  /-
    🎉 no goals
  -/


/-- Given an exact sequence `0 → M → N → P`, giving a section `P → N` is equivalent to giving a
splitting `N ≃ M × P`. -/
noncomputable
def Exact.splitSurjectiveEquiv (h : Function.Exact f g) (hf : Function.Injective f) :
    { l // g ∘ₗ l = .id } ≃
      { e : N ≃ₗ[R] M × P // f = e.symm ∘ₗ inl R M P ∧ g = snd R M P ∘ₗ e } := by
  refine
  { toFun := fun l ↦ ⟨(LinearEquiv.ofBijective (f ∘ₗ fst R M P + l.1 ∘ₗ snd R M P) ?_).symm, ?_⟩
    invFun := fun e ↦ ⟨e.1.symm ∘ₗ inr R M P, ?_⟩
    left_inv := ?_
    right_inv := ?_ }
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      l : Subtype fun l => Eq (g.comp l) LinearMap.id
      ⊢ Function.Bijective ⇑(HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (Li …
    -/
  · have h₁ : ∀ x, g (l.1 x) = x := LinearMap.congr_fun l.2
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      l : Subtype fun l => Eq (g.comp l) LinearMap.id
      h₁ : ∀ (x : P), Eq (g (↑l x)) x
      ⊢ Function.Bijective ⇑(HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (Li …
    -/
    have h₂ : ∀ x, g (f x) = 0 := congr_fun h.comp_eq_zero
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      l : Subtype fun l => Eq (g.comp l) LinearMap.id
      h₁ : ∀ (x : P), Eq (g (↑l x)) x
      h₂ : ∀ (x : M), Eq (g (f x)) 0
      ⊢ Function.Bijective ⇑(HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (Li …
    -/
    constructor
      /-
        case refine_1.left
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Function.Injective ⇑(HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (Li …
      -/
    · intros x y e
      /-
        case refine_1.left
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : Prod M P
        e : Eq ((HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (LinearMap.snd R  …
        ⊢ Eq x y
      -/
      simp only [add_apply, coe_comp, comp_apply, fst_apply, snd_apply] at e
      /-
        case refine_1.left
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : Prod M P
        e : Eq (HAdd.hAdd (f x.1) (↑l x.2)) (HAdd.hAdd (f y.1) (↑l y.2))
        ⊢ Eq x y
      -/
      suffices x.2 = y.2 from Prod.ext (hf (by rwa [this, add_left_inj] at e)) this
      /-
        case refine_1.left
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : Prod M P
        e : Eq (HAdd.hAdd (f x.1) (↑l x.2)) (HAdd.hAdd (f y.1) (↑l y.2))
        ⊢ Eq x.2 y.2
      -/
      simpa [h₁, h₂] using DFunLike.congr_arg g e
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Function.Surjective ⇑(HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (L …
      -/
    · intro x
      /-
        case refine_1.right
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x : N
        ⊢ Exists fun a => Eq ((HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (Li …
      -/
      obtain ⟨y, hy⟩ := (h (x - l.1 (g x))).mp (by simp [h₁, g.map_sub])
      /-
        case refine_1.right.intro
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x : N
        y : M
        hy : Eq (f y) (HSub.hSub x (↑l (g x)))
        ⊢ Exists fun a => Eq ((HAdd.hAdd (f.comp (LinearMap.fst R M P)) ((↑l).comp (Li …
      -/
      exact ⟨⟨y, g x⟩, by simp [hy]⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      l : Subtype fun l => Eq (g.comp l) LinearMap.id
      ⊢ And (Eq f ((↑(LinearEquiv.ofBijective (HAdd.hAdd (f.comp (LinearMap.fst R M  …
    -/
  · have h₁ : ∀ x, g (l.1 x) = x := LinearMap.congr_fun l.2
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      l : Subtype fun l => Eq (g.comp l) LinearMap.id
      h₁ : ∀ (x : P), Eq (g (↑l x)) x
      ⊢ And (Eq f ((↑(LinearEquiv.ofBijective (HAdd.hAdd (f.comp (LinearMap.fst R M  …
    -/
    have h₂ : ∀ x, g (f x) = 0 := congr_fun h.comp_eq_zero
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      l : Subtype fun l => Eq (g.comp l) LinearMap.id
      h₁ : ∀ (x : P), Eq (g (↑l x)) x
      h₂ : ∀ (x : M), Eq (g (f x)) 0
      ⊢ And (Eq f ((↑(LinearEquiv.ofBijective (HAdd.hAdd (f.comp (LinearMap.fst R M  …
    -/
    constructor
      /-
        case refine_2.left
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Eq f ((↑(LinearEquiv.ofBijective (HAdd.hAdd (f.comp (LinearMap.fst R M P)) ( …
      -/
    · ext; simp
           /-
             🎉 no goals
           -/
      /-
        case refine_2.right
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Eq g ((LinearMap.snd R M P).comp ↑(LinearEquiv.ofBijective (HAdd.hAdd (f.com …
      -/
    · rw [LinearEquiv.eq_comp_toLinearMap_symm]
      /-
        case refine_2.right
        R : Type u_1
        M : Type u_2
        M' : Type u_3
        N : Type u_4
        N' : Type u_5
        P : Type u_6
        P' : Type u_7
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hf : Function.Injective ⇑f
        l : Subtype fun l => Eq (g.comp l) LinearMap.id
        h₁ : ∀ (x : P), Eq (g (↑l x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Eq (g.comp ↑(LinearEquiv.ofBijective (HAdd.hAdd (f.comp (LinearMap.fst R M P …
      -/
              /-
                🎉 no goals
              -/
      ext <;> simp [h₁, h₂]
              /-
                🎉 no goals
              -/
    /-
      case refine_3
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      e : Subtype fun e => And (Eq f ((↑e.symm).comp (LinearMap.inl R M P))) (Eq g ( …
      ⊢ Eq (g.comp ((↑(↑e).symm).comp (LinearMap.inr R M P))) LinearMap.id
    -/
  · rw [← LinearMap.comp_assoc, (LinearEquiv.eq_comp_toLinearMap_symm _ _).mp e.2.2]; rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
    /-
      case refine_4
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      ⊢ Function.LeftInverse (fun e => ⟨(↑(↑e).symm).comp (LinearMap.inr R M P), ⋯⟩) …
    -/
  · intro; ext; simp
                /-
                  🎉 no goals
                -/
    /-
      case refine_5
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hf : Function.Injective ⇑f
      ⊢ Function.RightInverse (fun e => ⟨(↑(↑e).symm).comp (LinearMap.inr R M P), ⋯⟩ …
    -/
  · rintro ⟨e, rfl, rfl⟩
    /-
      case refine_5.mk.intro
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      e : LinearEquiv (RingHom.id R) N (Prod M P)
      hf : Function.Injective ⇑((↑e.symm).comp (LinearMap.inl R M P))
      h : Function.Exact ⇑((↑e.symm).comp (LinearMap.inl R M P)) ⇑((LinearMap.snd R  …
      ⊢ Eq ((fun l => ⟨(LinearEquiv.ofBijective (HAdd.hAdd (((↑e.symm).comp (LinearM …
    -/
    ext1
    /-
      case refine_5.mk.intro.a
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      e : LinearEquiv (RingHom.id R) N (Prod M P)
      hf : Function.Injective ⇑((↑e.symm).comp (LinearMap.inl R M P))
      h : Function.Exact ⇑((↑e.symm).comp (LinearMap.inl R M P)) ⇑((LinearMap.snd R  …
      ⊢ Eq ↑((fun l => ⟨(LinearEquiv.ofBijective (HAdd.hAdd (((↑e.symm).comp (Linear …
    -/
    apply LinearEquiv.symm_bijective.injective
    /-
      case refine_5.mk.intro.a.a
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      e : LinearEquiv (RingHom.id R) N (Prod M P)
      hf : Function.Injective ⇑((↑e.symm).comp (LinearMap.inl R M P))
      h : Function.Exact ⇑((↑e.symm).comp (LinearMap.inl R M P)) ⇑((LinearMap.snd R  …
      ⊢ Eq (↑((fun l => ⟨(LinearEquiv.ofBijective (HAdd.hAdd (((↑e.symm).comp (Linea …
    -/
    ext
    /-
      case refine_5.mk.intro.a.a.h
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      e : LinearEquiv (RingHom.id R) N (Prod M P)
      hf : Function.Injective ⇑((↑e.symm).comp (LinearMap.inl R M P))
      h : Function.Exact ⇑((↑e.symm).comp (LinearMap.inl R M P)) ⇑((LinearMap.snd R  …
      x✝ : Prod M P
      ⊢ Eq ((↑((fun l => ⟨(LinearEquiv.ofBijective (HAdd.hAdd (((↑e.symm).comp (Line …
    -/
    apply e.injective
    /-
      case refine_5.mk.intro.a.a.h.a
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      N : Type u_4
      N' : Type u_5
      P : Type u_6
      P' : Type u_7
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      e : LinearEquiv (RingHom.id R) N (Prod M P)
      hf : Function.Injective ⇑((↑e.symm).comp (LinearMap.inl R M P))
      h : Function.Exact ⇑((↑e.symm).comp (LinearMap.inl R M P)) ⇑((LinearMap.snd R  …
      x✝ : Prod M P
      ⊢ Eq (e ((↑((fun l => ⟨(LinearEquiv.ofBijective (HAdd.hAdd (((↑e.symm).comp (L …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp
            /-
              🎉 no goals
            -/


/-- Given an exact sequence `M → N → P → 0`, giving a retraction `N → M` is equivalent to giving a
splitting `N ≃ M × P`. -/
noncomputable
def Exact.splitInjectiveEquiv
    {R M N P} [Semiring R] [AddCommGroup M] [AddCommGroup N]
    [AddCommGroup P] [Module R M] [Module R N] [Module R P] {f : M →ₗ[R] N} {g : N →ₗ[R] P}
    (h : Function.Exact f g) (hg : Function.Surjective g) :
    { l // l ∘ₗ f = .id } ≃
      { e : N ≃ₗ[R] M × P // f = e.symm ∘ₗ inl R M P ∧ g = snd R M P ∘ₗ e } := by
  refine
  { toFun := fun l ↦ ⟨(LinearEquiv.ofBijective (l.1.prod g) ?_), ?_⟩
    invFun := fun e ↦ ⟨fst R M P ∘ₗ e.1, ?_⟩
    left_inv := ?_
    right_inv := ?_ }
    /-
      case refine_1
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      l : Subtype fun l => Eq (l.comp f) LinearMap.id
      ⊢ Function.Bijective ⇑((↑l).prod g)
    -/
  · have h₁ : ∀ x, l.1 (f x) = x := LinearMap.congr_fun l.2
    /-
      case refine_1
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      l : Subtype fun l => Eq (l.comp f) LinearMap.id
      h₁ : ∀ (x : M), Eq (↑l (f x)) x
      ⊢ Function.Bijective ⇑((↑l).prod g)
    -/
    have h₂ : ∀ x, g (f x) = 0 := congr_fun h.comp_eq_zero
    /-
      case refine_1
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      l : Subtype fun l => Eq (l.comp f) LinearMap.id
      h₁ : ∀ (x : M), Eq (↑l (f x)) x
      h₂ : ∀ (x : M), Eq (g (f x)) 0
      ⊢ Function.Bijective ⇑((↑l).prod g)
    -/
    constructor
      /-
        case refine_1.left
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Function.Injective ⇑((↑l).prod g)
      -/
    · intros x y e
      /-
        case refine_1.left
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : N
        e : Eq (((↑l).prod g) x) (((↑l).prod g) y)
        ⊢ Eq x y
      -/
      simp only [prod_apply, Pi.prod, Prod.mk.injEq] at e
      /-
        case refine_1.left
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : N
        e : And (Eq (↑l x) (↑l y)) (Eq (g x) (g y))
        ⊢ Eq x y
      -/
      obtain ⟨z, hz⟩ := (h (x - y)).mp (by simpa [sub_eq_zero] using e.2)
      /-
        case refine_1.left.intro
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : N
        e : And (Eq (↑l x) (↑l y)) (Eq (g x) (g y))
        z : M
        hz : Eq (f z) (HSub.hSub x y)
        ⊢ Eq x y
      -/
      suffices z = 0 by rw [← sub_eq_zero, ← hz, this, map_zero]
      /-
        case refine_1.left.intro
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x y : N
        e : And (Eq (↑l x) (↑l y)) (Eq (g x) (g y))
        z : M
        hz : Eq (f z) (HSub.hSub x y)
        ⊢ Eq z 0
      -/
      rw [← h₁ z, hz, map_sub, e.1, sub_self]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.right
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Function.Surjective ⇑((↑l).prod g)
      -/
    · rintro ⟨x, y⟩
      /-
        case refine_1.right.mk
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x : M
        y : P
        ⊢ Exists fun a => Eq (((↑l).prod g) a) { fst := x, snd := y }
      -/
      obtain ⟨y, rfl⟩ := hg y
      /-
        case refine_1.right.mk.intro
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        x : M
        y : N
        ⊢ Exists fun a => Eq (((↑l).prod g) a) { fst := x, snd := g y }
      -/
      refine ⟨f x + y - f (l.1 y), by ext <;> simp [h₁, h₂]⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      l : Subtype fun l => Eq (l.comp f) LinearMap.id
      ⊢ And (Eq f ((↑(LinearEquiv.ofBijective ((↑l).prod g) ⋯).symm).comp (LinearMap …
    -/
  · have h₁ : ∀ x, l.1 (f x) = x := LinearMap.congr_fun l.2
    /-
      case refine_2
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      l : Subtype fun l => Eq (l.comp f) LinearMap.id
      h₁ : ∀ (x : M), Eq (↑l (f x)) x
      ⊢ And (Eq f ((↑(LinearEquiv.ofBijective ((↑l).prod g) ⋯).symm).comp (LinearMap …
    -/
    have h₂ : ∀ x, g (f x) = 0 := congr_fun h.comp_eq_zero
    /-
      case refine_2
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      l : Subtype fun l => Eq (l.comp f) LinearMap.id
      h₁ : ∀ (x : M), Eq (↑l (f x)) x
      h₂ : ∀ (x : M), Eq (g (f x)) 0
      ⊢ And (Eq f ((↑(LinearEquiv.ofBijective ((↑l).prod g) ⋯).symm).comp (LinearMap …
    -/
    constructor
      /-
        case refine_2.left
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Eq f ((↑(LinearEquiv.ofBijective ((↑l).prod g) ⋯).symm).comp (LinearMap.inl  …
      -/
    · rw [LinearEquiv.eq_toLinearMap_symm_comp]
      /-
        case refine_2.left
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Eq ((↑(LinearEquiv.ofBijective ((↑l).prod g) ⋯)).comp f) (LinearMap.inl R M P)
      -/
              /-
                🎉 no goals
              -/
      ext <;> simp [h₁, h₂]
              /-
                🎉 no goals
              -/
      /-
        case refine_2.right
        R✝ : Type u_1
        M✝ : Type u_2
        M' : Type u_3
        N✝ : Type u_4
        N' : Type u_5
        P✝ : Type u_6
        P' : Type u_7
        inst✝¹³ : Semiring R✝
        inst✝¹² : AddCommGroup M✝
        inst✝¹¹ : AddCommGroup N✝
        inst✝¹⁰ : AddCommGroup P✝
        inst✝⁹ : Module R✝ M✝
        inst✝⁸ : Module R✝ N✝
        inst✝⁷ : Module R✝ P✝
        f✝ : LinearMap (RingHom.id R✝) M✝ N✝
        g✝ : LinearMap (RingHom.id R✝) N✝ P✝
        R : Type ?u.165039
        M : Type ?u.165042
        N : Type ?u.165045
        P : Type ?u.165048
        inst✝⁶ : Semiring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup N
        inst✝³ : AddCommGroup P
        inst✝² : Module R M
        inst✝¹ : Module R N
        inst✝ : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        h : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        l : Subtype fun l => Eq (l.comp f) LinearMap.id
        h₁ : ∀ (x : M), Eq (↑l (f x)) x
        h₂ : ∀ (x : M), Eq (g (f x)) 0
        ⊢ Eq g ((LinearMap.snd R M P).comp ↑(LinearEquiv.ofBijective ((↑l).prod g) ⋯))
      -/
    · ext; simp
           /-
             🎉 no goals
           -/
    /-
      case refine_3
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      e : Subtype fun e => And (Eq f ((↑e.symm).comp (LinearMap.inl R M P))) (Eq g ( …
      ⊢ Eq (((LinearMap.fst R M P).comp ↑↑e).comp f) LinearMap.id
    -/
  · rw [LinearMap.comp_assoc, (LinearEquiv.eq_toLinearMap_symm_comp _ _).mp e.2.1]; rfl
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/
    /-
      case refine_4
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      ⊢ Function.LeftInverse (fun e => ⟨(LinearMap.fst R M P).comp ↑↑e, ⋯⟩) fun l => …
    -/
  · intro; ext; simp
                /-
                  🎉 no goals
                -/
    /-
      case refine_5
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f✝ : LinearMap (RingHom.id R✝) M✝ N✝
      g✝ : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      ⊢ Function.RightInverse (fun e => ⟨(LinearMap.fst R M P).comp ↑↑e, ⋯⟩) fun l = …
    -/
  · rintro ⟨e, rfl, rfl⟩
    /-
      case refine_5.mk.intro
      R✝ : Type u_1
      M✝ : Type u_2
      M' : Type u_3
      N✝ : Type u_4
      N' : Type u_5
      P✝ : Type u_6
      P' : Type u_7
      inst✝¹³ : Semiring R✝
      inst✝¹² : AddCommGroup M✝
      inst✝¹¹ : AddCommGroup N✝
      inst✝¹⁰ : AddCommGroup P✝
      inst✝⁹ : Module R✝ M✝
      inst✝⁸ : Module R✝ N✝
      inst✝⁷ : Module R✝ P✝
      f : LinearMap (RingHom.id R✝) M✝ N✝
      g : LinearMap (RingHom.id R✝) N✝ P✝
      R : Type ?u.165039
      M : Type ?u.165042
      N : Type ?u.165045
      P : Type ?u.165048
      inst✝⁶ : Semiring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      e : LinearEquiv (RingHom.id R) N (Prod M P)
      hg : Function.Surjective ⇑((LinearMap.snd R M P).comp ↑e)
      h : Function.Exact ⇑((↑e.symm).comp (LinearMap.inl R M P)) ⇑((LinearMap.snd R  …
      ⊢ Eq ((fun l => ⟨LinearEquiv.ofBijective ((↑l).prod ((LinearMap.snd R M P).com …
    -/
              /-
                🎉 no goals
              -/
    ext x <;> simp
              /-
                🎉 no goals
              -/


theorem Exact.split_tfae' (h : Function.Exact f g) :
    List.TFAE [
      Function.Injective f ∧ ∃ l, g ∘ₗ l = LinearMap.id,
      Function.Surjective g ∧ ∃ l, l ∘ₗ f = LinearMap.id,
      ∃ e : N ≃ₗ[R] M × P, f = e.symm ∘ₗ LinearMap.inl R M P ∧ g = LinearMap.snd R M P ∘ₗ e] := by
  tfae_have 1 → 3
  | ⟨hf, l, hl⟩ => ⟨_, (h.splitSurjectiveEquiv hf ⟨l, hl⟩).2⟩
  tfae_have 2 → 3
  | ⟨hg, l, hl⟩ => ⟨_, (h.splitInjectiveEquiv hg ⟨l, hl⟩).2⟩
  tfae_have 3 → 1
  | ⟨e, e₁, e₂⟩ => by
    have : Function.Injective f := e₁ ▸ e.symm.injective.comp LinearMap.inl_injective
    exact ⟨this, ⟨_, ((h.splitSurjectiveEquiv this).symm ⟨e, e₁, e₂⟩).2⟩⟩
  tfae_have 3 → 2
  | ⟨e, e₁, e₂⟩ => by
    have : Function.Surjective g := e₂ ▸ Prod.snd_surjective.comp e.surjective
    exact ⟨this, ⟨_, ((h.splitInjectiveEquiv this).symm ⟨e, e₁, e₂⟩).2⟩⟩
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    tfae_1_to_3 : And (Function.Injective ⇑f) (Exists fun l => Eq (g.comp l) Linea …
    tfae_2_to_3 : And (Function.Surjective ⇑g) (Exists fun l => Eq (l.comp f) Line …
    tfae_3_to_1 : (Exists fun e => And (Eq f ((↑e.symm).comp (LinearMap.inl R M P) …
    tfae_3_to_2 : (Exists fun e => And (Eq f ((↑e.symm).comp (LinearMap.inl R M P) …
    ⊢ (List.cons (And (Function.Injective ⇑f) (Exists fun l => Eq (g.comp l) Linea …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- Equivalent characterizations of split exact sequences. Also known as the **Splitting lemma**. -/
theorem Exact.split_tfae
    {R M N P} [Semiring R] [AddCommGroup M] [AddCommGroup N]
    [AddCommGroup P] [Module R M] [Module R N] [Module R P] {f : M →ₗ[R] N} {g : N →ₗ[R] P}
    (h : Function.Exact f g) (hf : Function.Injective f) (hg : Function.Surjective g) :
    List.TFAE [
      ∃ l, g ∘ₗ l = LinearMap.id,
      ∃ l, l ∘ₗ f = LinearMap.id,
      ∃ e : N ≃ₗ[R] M × P, f = e.symm ∘ₗ LinearMap.inl R M P ∧ g = LinearMap.snd R M P ∘ₗ e] := by
  tfae_have 1 ↔ 3 := by
    simpa using (h.splitSurjectiveEquiv hf).nonempty_congr
  tfae_have 2 ↔ 3 := by
    simpa using (h.splitInjectiveEquiv hg).nonempty_congr
  /-
    R : Type u_8
    M : Type u_9
    N : Type u_10
    P : Type u_11
    inst✝⁶ : Semiring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    hf : Function.Injective ⇑f
    hg : Function.Surjective ⇑g
    tfae_1_iff_3 : Iff (Exists fun l => Eq (g.comp l) LinearMap.id) (Exists fun e  …
    tfae_2_iff_3 : Iff (Exists fun l => Eq (l.comp f) LinearMap.id) (Exists fun e  …
    ⊢ (List.cons (Exists fun l => Eq (g.comp l) LinearMap.id) (List.cons (Exists f …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma Exact.inr_fst : Function.Exact (LinearMap.inr R M N) (LinearMap.fst R M N) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ Function.Exact ⇑(LinearMap.inr R M N) ⇑(LinearMap.fst R M N)
  -/
  rintro ⟨x, y⟩
  simp only [LinearMap.fst_apply, @eq_comm _ x, LinearMap.coe_inr, Set.mem_range, Prod.mk.injEq,
    exists_eq_right]


lemma Exact.inl_snd : Function.Exact (LinearMap.inl R M N) (LinearMap.snd R M N) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid N
    inst✝¹ : Module R M
    inst✝ : Module R N
    ⊢ Function.Exact ⇑(LinearMap.inl R M N) ⇑(LinearMap.snd R M N)
  -/
  rintro ⟨x, y⟩
  simp only [LinearMap.snd_apply, @eq_comm _ y, LinearMap.coe_inl, Set.mem_range, Prod.mk.injEq,
    exists_eq_left]


/-- A necessary and sufficient condition for an exact sequence to descend to a quotient. -/
lemma Exact.exact_mapQ_iff
    (hfg : Exact f g) {p q r} (hpq : p ≤ comap f q) (hqr : q ≤ comap g r) :
    Exact (mapQ p q f hpq) (mapQ q r g hqr) ↔ range g ⊓ r ≤ map g q := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    p : Submodule R M
    q : Submodule R N
    r : Submodule R P
    hpq : LE.le p (Submodule.comap f q)
    hqr : LE.le q (Submodule.comap g r)
    ⊢ Iff (Function.Exact ⇑(p.mapQ q f hpq) ⇑(q.mapQ r g hqr)) (LE.le (Min.min (Li …
  -/
  rw [exact_iff, ← (comap_injective_of_surjective (mkQ_surjective _)).eq_iff]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    p : Submodule R M
    q : Submodule R N
    r : Submodule R P
    hpq : LE.le p (Submodule.comap f q)
    hqr : LE.le q (Submodule.comap g r)
    ⊢ Iff (Eq (Submodule.comap q.mkQ (LinearMap.ker (q.mapQ r g hqr))) (Submodule. …
  -/
  dsimp only [mapQ]
  rw [← ker_comp, range_liftQ, liftQ_mkQ, ker_comp, range_comp, comap_map_eq,
    ker_mkQ, ker_mkQ, ← hfg.linearMap_ker_eq, sup_comm,
    ← LE.le.le_iff_eq (sup_le hqr (ker_le_comap g)),
    ← comap_map_eq, ← map_le_iff_le_comap, map_comap_eq]


/-- When we have a commutative diagram from a sequence of two linear maps to another,
such that the left vertical map is surjective, the middle vertical map is bijective and the right
vertical map is injective, then the upper row is exact iff the lower row is.
See `ShortComplex.exact_iff_of_epi_of_isIso_of_mono` in the file
`Algebra.Homology.ShortComplex.Exact` for the categorical version of this result. -/
lemma exact_iff_of_surjective_of_bijective_of_injective
  {M₁ M₂ M₃ N₁ N₂ N₃ : Type*} [AddCommMonoid M₁] [AddCommMonoid M₂] [AddCommMonoid M₃]
  [AddCommMonoid N₁] [AddCommMonoid N₂] [AddCommMonoid N₃]
  [Module R M₁] [Module R M₂] [Module R M₃]
  [Module R N₁] [Module R N₂] [Module R N₃]
  (f : M₁ →ₗ[R] M₂) (g : M₂ →ₗ[R] M₃) (f' : N₁ →ₗ[R] N₂) (g' : N₂ →ₗ[R] N₃)
  (τ₁ : M₁ →ₗ[R] N₁) (τ₂ : M₂ →ₗ[R] N₂) (τ₃ : M₃ →ₗ[R] N₃)
  (comm₁₂ : f'.comp τ₁ = τ₂.comp f) (comm₂₃ : g'.comp τ₂ = τ₃.comp g)
  (h₁ : Function.Surjective τ₁) (h₂ : Function.Bijective τ₂) (h₃ : Function.Injective τ₃) :
    Function.Exact f g ↔ Function.Exact f' g' :=
  AddMonoidHom.exact_iff_of_surjective_of_bijective_of_injective
    f.toAddMonoidHom g.toAddMonoidHom f'.toAddMonoidHom g'.toAddMonoidHom
    τ₁.toAddMonoidHom τ₂.toAddMonoidHom τ₃.toAddMonoidHom
        /-
          R : Type u_1
          inst✝¹² : Ring R
          M₁ : Type u_8
          M₂ : Type u_9
          M₃ : Type u_10
          N₁ : Type u_11
          N₂ : Type u_12
          N₃ : Type u_13
          inst✝¹¹ : AddCommMonoid M₁
          inst✝¹⁰ : AddCommMonoid M₂
          inst✝⁹ : AddCommMonoid M₃
          inst✝⁸ : AddCommMonoid N₁
          inst✝⁷ : AddCommMonoid N₂
          inst✝⁶ : AddCommMonoid N₃
          inst✝⁵ : Module R M₁
          inst✝⁴ : Module R M₂
          inst✝³ : Module R M₃
          inst✝² : Module R N₁
          inst✝¹ : Module R N₂
          inst✝ : Module R N₃
          f : LinearMap (RingHom.id R) M₁ M₂
          g : LinearMap (RingHom.id R) M₂ M₃
          f' : LinearMap (RingHom.id R) N₁ N₂
          g' : LinearMap (RingHom.id R) N₂ N₃
          τ₁ : LinearMap (RingHom.id R) M₁ N₁
          τ₂ : LinearMap (RingHom.id R) M₂ N₂
          τ₃ : LinearMap (RingHom.id R) M₃ N₃
          comm₁₂ : Eq (f'.comp τ₁) (τ₂.comp f)
          comm₂₃ : Eq (g'.comp τ₂) (τ₃.comp g)
          h₁ : Function.Surjective ⇑τ₁
          h₂ : Function.Bijective ⇑τ₂
          h₃ : Function.Injective ⇑τ₃
          ⊢ Eq (f'.toAddMonoidHom.comp τ₁.toAddMonoidHom) (τ₂.toAddMonoidHom.comp f.toAd …
        -/
             /-
               🎉 no goals
             -/
    (by ext; apply DFunLike.congr_fun comm₁₂) (by ext; apply DFunLike.congr_fun comm₂₃) h₁ h₂ h₃
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma surjective_range_liftQ (h : range f ≤ ker g) (hg : Function.Surjective g) :
    Function.Surjective ((range f).liftQ g h) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LE.le (LinearMap.range f) (LinearMap.ker g)
    hg : Function.Surjective ⇑g
    ⊢ Function.Surjective ⇑((LinearMap.range f).liftQ g h)
  -/
  intro x₃
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LE.le (LinearMap.range f) (LinearMap.ker g)
    hg : Function.Surjective ⇑g
    x₃ : P
    ⊢ Exists fun a => Eq (((LinearMap.range f).liftQ g h) a) x₃
  -/
  obtain ⟨x₂, rfl⟩ := hg x₃
  /-
    case intro
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LE.le (LinearMap.range f) (LinearMap.ker g)
    hg : Function.Surjective ⇑g
    x₂ : N
    ⊢ Exists fun a => Eq (((LinearMap.range f).liftQ g h) a) (g x₂)
  -/
  exact ⟨Submodule.Quotient.mk x₂, rfl⟩
  /-
    🎉 no goals
  -/


lemma ker_eq_bot_range_liftQ_iff (h : range f ≤ ker g) :
    ker ((range f).liftQ g h) = ⊥ ↔ ker g = range f := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LE.le (LinearMap.range f) (LinearMap.ker g)
    ⊢ Iff (Eq (LinearMap.ker ((LinearMap.range f).liftQ g h)) Bot.bot) (Eq (Linear …
  -/
  simp only [Submodule.ext_iff, mem_ker, Submodule.mem_bot, mem_range]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LE.le (LinearMap.range f) (LinearMap.ker g)
    ⊢ Iff (∀ (x : HasQuotient.Quotient N (LinearMap.range f)), Iff (Eq (((LinearMa …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      N : Type u_4
      P : Type u_6
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : LE.le (LinearMap.range f) (LinearMap.ker g)
      ⊢ (∀ (x : HasQuotient.Quotient N (LinearMap.range f)), Iff (Eq (((LinearMap.ra …
    -/
  · intro hfg x
    /-
      case mp
      R : Type u_1
      M : Type u_2
      N : Type u_4
      P : Type u_6
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : LE.le (LinearMap.range f) (LinearMap.ker g)
      hfg : ∀ (x : HasQuotient.Quotient N (LinearMap.range f)), Iff (Eq (((LinearMap …
      x : N
      ⊢ Iff (Eq (g x) 0) (Exists fun y => Eq (f y) x)
    -/
    simpa using hfg (Submodule.Quotient.mk x)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      N : Type u_4
      P : Type u_6
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : LE.le (LinearMap.range f) (LinearMap.ker g)
      ⊢ (∀ (x : N), Iff (Eq (g x) 0) (Exists fun y => Eq (f y) x)) → ∀ (x : HasQuoti …
    -/
  · intro hfg x
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      N : Type u_4
      P : Type u_6
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : LE.le (LinearMap.range f) (LinearMap.ker g)
      hfg : ∀ (x : N), Iff (Eq (g x) 0) (Exists fun y => Eq (f y) x)
      x : HasQuotient.Quotient N (LinearMap.range f)
      ⊢ Iff (Eq (((LinearMap.range f).liftQ g h) x) 0) (Eq x 0)
    -/
    obtain ⟨x, rfl⟩ := Submodule.Quotient.mk_surjective _ x
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      N : Type u_4
      P : Type u_6
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : AddCommGroup N
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R N
      inst✝ : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      h : LE.le (LinearMap.range f) (LinearMap.ker g)
      hfg : ∀ (x : N), Iff (Eq (g x) 0) (Exists fun y => Eq (f y) x)
      x : N
      ⊢ Iff (Eq (((LinearMap.range f).liftQ g h) (Submodule.Quotient.mk x)) 0) (Eq ( …
    -/
    simpa using hfg x
    /-
      🎉 no goals
    -/


lemma injective_range_liftQ_of_exact (h : Function.Exact f g) :
    Function.Injective ((range f).liftQ g (h · |>.mpr)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_4
    P : Type u_6
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : Function.Exact ⇑f ⇑g
    ⊢ Function.Injective ⇑((LinearMap.range f).liftQ g ⋯)
  -/
  simpa only [← LinearMap.ker_eq_bot, ker_eq_bot_range_liftQ_iff, exact_iff] using h
  /-
    🎉 no goals
  -/


/-- The linear equivalence `(N ⧸ LinearMap.range f) ≃ₗ[A] P` associated to
an exact sequence `M → N → P → 0` of `R`-modules. -/
@[simps! apply]
noncomputable def Function.Exact.linearEquivOfSurjective (h : Function.Exact f g)
    (hg : Function.Surjective g) : (N ⧸ LinearMap.range f) ≃ₗ[R] P :=
  LinearEquiv.ofBijective ((LinearMap.range f).liftQ g (h · |>.mpr))
      ⟨LinearMap.injective_range_liftQ_of_exact h,
        LinearMap.surjective_range_liftQ _ hg⟩


