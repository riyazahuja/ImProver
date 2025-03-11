/-- The canonical map from the disjoint union induced by `f` to `S`. -/
@[simps apply]
def sigmaIsoHom : C((x : Fiber f) × x.val, S) where
  toFun | ⟨a, x⟩ => x.val


lemma sigmaIsoHom_inj : Function.Injective (sigmaIsoHom f) := by
  /-
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝ : TopologicalSpace S
    ⊢ Function.Injective ⇑(TopologicalSpace.Fiber.sigmaIsoHom f)
  -/
  rintro ⟨⟨_, _, rfl⟩, ⟨_, hx⟩⟩ ⟨⟨_, _, rfl⟩, ⟨_, hy⟩⟩ h
  /-
    case mk.mk.intro.mk.mk.mk.intro.mk
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝ : TopologicalSpace S
    w✝¹ : ↑(Set.range f)
    val✝¹ : S
    hx : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, …
    w✝ : ↑(Set.range f)
    val✝ : S
    hy : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝,  …
    h : Eq ((TopologicalSpace.Fiber.sigmaIsoHom f) ⟨⟨(fun x => Set.preimage f (Sin …
    ⊢ Eq ⟨⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, ⋯⟩, ⟨val✝¹, hx⟩ …
  -/
  refine Sigma.subtype_ext ?_ h
  /-
    case mk.mk.intro.mk.mk.mk.intro.mk
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝ : TopologicalSpace S
    w✝¹ : ↑(Set.range f)
    val✝¹ : S
    hx : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, …
    w✝ : ↑(Set.range f)
    val✝ : S
    hy : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝,  …
    h : Eq ((TopologicalSpace.Fiber.sigmaIsoHom f) ⟨⟨(fun x => Set.preimage f (Sin …
    ⊢ Eq ⟨⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, ⋯⟩, ⟨val✝¹, hx⟩ …
  -/
  simp only [sigmaIsoHom_apply] at h
  /-
    case mk.mk.intro.mk.mk.mk.intro.mk
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝ : TopologicalSpace S
    w✝¹ : ↑(Set.range f)
    val✝¹ : S
    hx : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, …
    w✝ : ↑(Set.range f)
    val✝ : S
    hy : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝,  …
    h : Eq val✝¹ val✝
    ⊢ Eq ⟨⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, ⋯⟩, ⟨val✝¹, hx⟩ …
  -/
  rw [Set.mem_preimage, Set.mem_singleton_iff] at hx hy
  /-
    case mk.mk.intro.mk.mk.mk.intro.mk
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝ : TopologicalSpace S
    w✝¹ : ↑(Set.range f)
    val✝¹ : S
    hx✝ : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹ …
    hx : Eq (f val✝¹) ↑w✝¹
    w✝ : ↑(Set.range f)
    val✝ : S
    hy✝ : Membership.mem (↑⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝, …
    hy : Eq (f val✝) ↑w✝
    h : Eq val✝¹ val✝
    ⊢ Eq ⟨⟨(fun x => Set.preimage f (Singleton.singleton ↑x)) w✝¹, ⋯⟩, ⟨val✝¹, hx✝ …
  -/
  simp [← hx, ← hy, h]
  /-
    🎉 no goals
  -/


lemma sigmaIsoHom_surj : Function.Surjective (sigmaIsoHom f) :=
  fun _ ↦ ⟨⟨⟨_, ⟨⟨_, Set.mem_range_self _⟩, rfl⟩⟩, ⟨_, rfl⟩⟩, rfl⟩


/-- The inclusion map from a component of the disjoint union induced by `f` into `S`. -/
def sigmaIncl (a : Fiber f) : C(a.val, S) where
  toFun x := x.val


/-- The inclusion map from a fiber of a composition into the intermediate fiber. -/
def sigmaInclIncl {X : Type*} (g : Y → X) (a : Fiber (g ∘ f))
    (b : Fiber (f ∘ (sigmaIncl (g ∘ f) a))) :
    C(b.val, (Fiber.mk f (b.preimage).val).val) where
  toFun x := ⟨x.val.val, by
    /-
      S : Type u_1
      Y : Type u_2
      f : S → Y
      inst✝ : TopologicalSpace S
      X : Type u_3
      g : Y → X
      a : Function.Fiber (Function.comp g f)
      b : Function.Fiber (Function.comp f ⇑(TopologicalSpace.Fiber.sigmaIncl (Functi …
      x : ↑↑b
      ⊢ Membership.mem ↑(Function.Fiber.mk f ↑(Function.Fiber.preimage (Function.com …
    -/
    have := x.prop
    /-
      S : Type u_1
      Y : Type u_2
      f : S → Y
      inst✝ : TopologicalSpace S
      X : Type u_3
      g : Y → X
      a : Function.Fiber (Function.comp g f)
      b : Function.Fiber (Function.comp f ⇑(TopologicalSpace.Fiber.sigmaIncl (Functi …
      x : ↑↑b
      this : Membership.mem ↑b ↑x
      ⊢ Membership.mem ↑(Function.Fiber.mk f ↑(Function.Fiber.preimage (Function.com …
    -/
    simp only [sigmaIncl, ContinuousMap.coe_mk, Fiber.mem_iff_eq_image, comp_apply] at this
    /-
      S : Type u_1
      Y : Type u_2
      f : S → Y
      inst✝ : TopologicalSpace S
      X : Type u_3
      g : Y → X
      a : Function.Fiber (Function.comp g f)
      b : Function.Fiber (Function.comp f ⇑(TopologicalSpace.Fiber.sigmaIncl (Functi …
      x : ↑↑b
      this : Eq (f ↑↑x) (Function.Fiber.image (Function.comp f fun x => ↑x) b)
      ⊢ Membership.mem ↑(Function.Fiber.mk f ↑(Function.Fiber.preimage (Function.com …
    -/
    rw [Fiber.mem_iff_eq_image, Fiber.mk_image, this, ← Fiber.map_preimage_eq_image]
    /-
      S : Type u_1
      Y : Type u_2
      f : S → Y
      inst✝ : TopologicalSpace S
      X : Type u_3
      g : Y → X
      a : Function.Fiber (Function.comp g f)
      b : Function.Fiber (Function.comp f ⇑(TopologicalSpace.Fiber.sigmaIncl (Functi …
      x : ↑↑b
      this : Eq (f ↑↑x) (Function.Fiber.image (Function.comp f fun x => ↑x) b)
      ⊢ Eq (Function.comp f (fun x => ↑x) (Function.Fiber.preimage (Function.comp f  …
    -/
    simp [sigmaIncl]⟩
    /-
      🎉 no goals
    -/


instance (x : Fiber l) : CompactSpace x.val := by
  /-
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝¹ : TopologicalSpace S
    l : LocallyConstant S Y
    inst✝ : CompactSpace S
    x : Function.Fiber ⇑l
    ⊢ CompactSpace ↑↑x
  -/
  obtain ⟨y, hy⟩ := x.prop
  /-
    case intro
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝¹ : TopologicalSpace S
    l : LocallyConstant S Y
    inst✝ : CompactSpace S
    x : Function.Fiber ⇑l
    y : ↑(Set.range ⇑l)
    hy : Eq ((fun x => Set.preimage (⇑l) (Singleton.singleton ↑x)) y) ↑x
    ⊢ CompactSpace ↑↑x
  -/
  rw [← isCompact_iff_compactSpace, ← hy]
  /-
    case intro
    S : Type u_1
    Y : Type u_2
    f : S → Y
    inst✝¹ : TopologicalSpace S
    l : LocallyConstant S Y
    inst✝ : CompactSpace S
    x : Function.Fiber ⇑l
    y : ↑(Set.range ⇑l)
    hy : Eq ((fun x => Set.preimage (⇑l) (Singleton.singleton ↑x)) y) ↑x
    ⊢ IsCompact ((fun x => Set.preimage (⇑l) (Singleton.singleton ↑x)) y)
  -/
  exact (l.2.isClosed_fiber _).isCompact
  /-
    🎉 no goals
  -/


instance : Finite (Fiber l) :=
  have : Finite (Set.range l) := l.range_finite
  Finite.Set.finite_range _


