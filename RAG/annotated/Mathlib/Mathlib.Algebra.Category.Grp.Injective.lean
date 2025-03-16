theorem Module.Baer.of_divisible [DivisibleBy A ℤ] : Module.Baer ℤ A := fun I g ↦ by
  /-
    A : Type u
    inst✝¹ : AddCommGroup A
    inst✝ : DivisibleBy A Int
    I : Ideal Int
    g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem I x) A
    ⊢ Exists fun g' => ∀ (x : Int) (mem : Membership.mem I x), Eq (g' x) (g ⟨x, me …
  -/
  rcases IsPrincipalIdealRing.principal I with ⟨m, rfl⟩
  /-
    case mk.intro
    A : Type u
    inst✝¹ : AddCommGroup A
    inst✝ : DivisibleBy A Int
    m : Int
    g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
    ⊢ Exists fun g' => ∀ (x : Int) (mem : Membership.mem (Submodule.span Int (Sing …
  -/
  obtain rfl | h0 := eq_or_ne m 0
    /-
      case mk.intro.inl
      A : Type u
      inst✝¹ : AddCommGroup A
      inst✝ : DivisibleBy A Int
      g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
      ⊢ Exists fun g' => ∀ (x : Int) (mem : Membership.mem (Submodule.span Int (Sing …
    -/
  · refine ⟨0, fun n hn ↦ ?_⟩
    /-
      case mk.intro.inl
      A : Type u
      inst✝¹ : AddCommGroup A
      inst✝ : DivisibleBy A Int
      g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
      n : Int
      hn : Membership.mem (Submodule.span Int (Singleton.singleton 0)) n
      ⊢ Eq (0 n) (g ⟨n, hn⟩)
    -/
    rw [Submodule.span_zero_singleton] at hn
    /-
      case mk.intro.inl
      A : Type u
      inst✝¹ : AddCommGroup A
      inst✝ : DivisibleBy A Int
      g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
      n : Int
      hn✝ : Membership.mem (Submodule.span Int (Singleton.singleton 0)) n
      hn : Membership.mem Bot.bot n
      ⊢ Eq (0 n) (g ⟨n, hn✝⟩)
    -/
    subst hn
    /-
      case mk.intro.inl
      A : Type u
      inst✝¹ : AddCommGroup A
      inst✝ : DivisibleBy A Int
      g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
      hn : Membership.mem (Submodule.span Int (Singleton.singleton 0)) 0
      ⊢ Eq (0 0) (g ⟨0, hn⟩)
    -/
    exact (map_zero g).symm
    /-
      🎉 no goals
    -/
  /-
    case mk.intro.inr
    A : Type u
    inst✝¹ : AddCommGroup A
    inst✝ : DivisibleBy A Int
    m : Int
    g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
    h0 : Ne m 0
    ⊢ Exists fun g' => ∀ (x : Int) (mem : Membership.mem (Submodule.span Int (Sing …
  -/
  let gₘ := g ⟨m, Submodule.subset_span (Set.mem_singleton _)⟩
  /-
    case mk.intro.inr
    A : Type u
    inst✝¹ : AddCommGroup A
    inst✝ : DivisibleBy A Int
    m : Int
    g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
    h0 : Ne m 0
    gₘ : A := g ⟨m, ⋯⟩
    ⊢ Exists fun g' => ∀ (x : Int) (mem : Membership.mem (Submodule.span Int (Sing …
  -/
  refine ⟨LinearMap.toSpanSingleton ℤ A (DivisibleBy.div gₘ m), fun n hn ↦ ?_⟩
  /-
    case mk.intro.inr
    A : Type u
    inst✝¹ : AddCommGroup A
    inst✝ : DivisibleBy A Int
    m : Int
    g : LinearMap (RingHom.id Int) (Subtype fun x => Membership.mem (Submodule.spa …
    h0 : Ne m 0
    gₘ : A := g ⟨m, ⋯⟩
    n : Int
    hn : Membership.mem (Submodule.span Int (Singleton.singleton m)) n
    ⊢ Eq ((LinearMap.toSpanSingleton Int A (DivisibleBy.div gₘ m)) n) (g ⟨n, hn⟩)
  -/
  rcases Submodule.mem_span_singleton.mp hn with ⟨n, rfl⟩
  rw [map_zsmul, LinearMap.toSpanSingleton_apply, DivisibleBy.div_cancel gₘ h0, ← map_zsmul g,
    SetLike.mk_smul_mk]


theorem injective_as_module_iff : Injective (ModuleCat.of ℤ A) ↔
    Injective (⟨A,inferInstance⟩ : AddCommGrp) :=
  ((forget₂ (ModuleCat ℤ) AddCommGrp).asEquivalence.map_injective_iff (ModuleCat.of ℤ A)).symm


instance injective_of_divisible [DivisibleBy A ℤ] :
    Injective (⟨A,inferInstance⟩ : AddCommGrp) :=
  (injective_as_module_iff A).mp <|
    Module.injective_object_of_injective_module (inj := (Module.Baer.of_divisible A).injective)


instance injective_ratCircle : Injective <| of <| ULift.{u} <| AddCircle (1 : ℚ) :=
  injective_of_divisible _


