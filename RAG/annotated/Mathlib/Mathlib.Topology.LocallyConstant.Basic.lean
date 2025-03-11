/-- A function between topological spaces is locally constant if the preimage of any set is open. -/
def IsLocallyConstant (f : X → Y) : Prop :=
  ∀ s : Set Y, IsOpen (f ⁻¹' s)


open List in
protected theorem tfae (f : X → Y) :
    TFAE [IsLocallyConstant f,
      ∀ x, ∀ᶠ x' in 𝓝 x, f x' = f x,
      ∀ x, IsOpen { x' | f x' = f x },
      ∀ y, IsOpen (f ⁻¹' {y}),
      ∀ x, ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ ∀ x' ∈ U, f x' = f x] := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ (List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventually (f …
  -/
  tfae_have 1 → 4 := fun h y => h {y}
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    tfae_1_to_4 : IsLocallyConstant f → ∀ (y : Y), IsOpen (Set.preimage f (Singlet …
    ⊢ (List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventually (f …
  -/
  tfae_have 4 → 3 := fun h x => h (f x)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    tfae_1_to_4 : IsLocallyConstant f → ∀ (y : Y), IsOpen (Set.preimage f (Singlet …
    tfae_4_to_3 : (∀ (y : Y), IsOpen (Set.preimage f (Singleton.singleton y))) → ∀ …
    ⊢ (List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventually (f …
  -/
  tfae_have 3 → 2 := fun h x => IsOpen.mem_nhds (h x) rfl
  tfae_have 2 → 5
  | h, x => by
    rcases mem_nhds_iff.1 (h x) with ⟨U, eq, hU, hx⟩
    exact ⟨U, hU, hx, eq⟩
  tfae_have 5 → 1
  | h, s => by
    refine isOpen_iff_forall_mem_open.2 fun x hx ↦ ?_
    rcases h x with ⟨U, hU, hxU, eq⟩
    exact ⟨U, fun x' hx' => mem_preimage.2 <| (eq x' hx').symm ▸ hx, hU, hxU⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    tfae_1_to_4 : IsLocallyConstant f → ∀ (y : Y), IsOpen (Set.preimage f (Singlet …
    tfae_4_to_3 : (∀ (y : Y), IsOpen (Set.preimage f (Singleton.singleton y))) → ∀ …
    tfae_3_to_2 : (∀ (x : X), IsOpen (setOf fun x' => Eq (f x') (f x))) → ∀ (x : X …
    tfae_2_to_5 : (∀ (x : X), Filter.Eventually (fun x' => Eq (f x') (f x)) (nhds  …
    tfae_5_to_1 : (∀ (x : X), Exists fun U => And (IsOpen U) (And (Membership.mem  …
    ⊢ (List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventually (f …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


@[nontriviality]
theorem of_discrete [DiscreteTopology X] (f : X → Y) : IsLocallyConstant f := fun _ =>
  isOpen_discrete _


theorem isOpen_fiber {f : X → Y} (hf : IsLocallyConstant f) (y : Y) : IsOpen { x | f x = y } :=
  hf {y}


theorem isClosed_fiber {f : X → Y} (hf : IsLocallyConstant f) (y : Y) : IsClosed { x | f x = y } :=
  ⟨hf {y}ᶜ⟩


theorem isClopen_fiber {f : X → Y} (hf : IsLocallyConstant f) (y : Y) : IsClopen { x | f x = y } :=
  ⟨isClosed_fiber hf _,  isOpen_fiber hf _⟩


theorem iff_exists_open (f : X → Y) :
    IsLocallyConstant f ↔ ∀ x, ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ ∀ x' ∈ U, f x' = f x :=
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Eq ((List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventuall …
  -/
  /-
    🎉 no goals
  -/
  (IsLocallyConstant.tfae f).out 0 4
  /-
    🎉 no goals
  -/


theorem iff_eventually_eq (f : X → Y) : IsLocallyConstant f ↔ ∀ x, ∀ᶠ y in 𝓝 x, f y = f x :=
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Eq ((List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventuall …
  -/
  /-
    🎉 no goals
  -/
  (IsLocallyConstant.tfae f).out 0 1
  /-
    🎉 no goals
  -/


theorem exists_open {f : X → Y} (hf : IsLocallyConstant f) (x : X) :
    ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ ∀ x' ∈ U, f x' = f x :=
  (iff_exists_open f).1 hf x


protected theorem eventually_eq {f : X → Y} (hf : IsLocallyConstant f) (x : X) :
    ∀ᶠ y in 𝓝 x, f y = f x :=
  (iff_eventually_eq f).1 hf x


theorem iff_isOpen_fiber_apply {f : X → Y} : IsLocallyConstant f ↔ ∀ x, IsOpen (f ⁻¹' {f x}) :=
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Eq ((List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventuall …
  -/
  /-
    🎉 no goals
  -/
  (IsLocallyConstant.tfae f).out 0 2
  /-
    🎉 no goals
  -/


theorem iff_isOpen_fiber {f : X → Y} : IsLocallyConstant f ↔ ∀ y, IsOpen (f ⁻¹' {y}) :=
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Eq ((List.cons (IsLocallyConstant f) (List.cons (∀ (x : X), Filter.Eventuall …
  -/
  /-
    🎉 no goals
  -/
  (IsLocallyConstant.tfae f).out 0 3
  /-
    🎉 no goals
  -/


protected theorem continuous [TopologicalSpace Y] {f : X → Y} (hf : IsLocallyConstant f) :
    Continuous f :=
  ⟨fun _ _ => hf _⟩


theorem iff_continuous {_ : TopologicalSpace Y} [DiscreteTopology Y] (f : X → Y) :
    IsLocallyConstant f ↔ Continuous f :=
  ⟨IsLocallyConstant.continuous, fun h s => h.isOpen_preimage s (isOpen_discrete _)⟩


theorem of_constant (f : X → Y) (h : ∀ x y, f x = f y) : IsLocallyConstant f :=
  (iff_eventually_eq f).2 fun _ => Eventually.of_forall fun _ => h _ _


protected theorem const (y : Y) : IsLocallyConstant (Function.const X y) :=
  of_constant _ fun _ _ => rfl


protected theorem comp {f : X → Y} (hf : IsLocallyConstant f) (g : Y → Z) :
    IsLocallyConstant (g ∘ f) := fun s => by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝ : TopologicalSpace X
    f : X → Y
    hf : IsLocallyConstant f
    g : Y → Z
    s : Set Z
    ⊢ IsOpen (Set.preimage (Function.comp g f) s)
  -/
  rw [Set.preimage_comp]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝ : TopologicalSpace X
    f : X → Y
    hf : IsLocallyConstant f
    g : Y → Z
    s : Set Z
    ⊢ IsOpen (Set.preimage f (Set.preimage g s))
  -/
  exact hf _
  /-
    🎉 no goals
  -/


theorem prod_mk {Y'} {f : X → Y} {f' : X → Y'} (hf : IsLocallyConstant f)
    (hf' : IsLocallyConstant f') : IsLocallyConstant fun x => (f x, f' x) :=
  (iff_eventually_eq _).2 fun x =>
    (hf.eventually_eq x).mp <| (hf'.eventually_eq x).mono fun _ hf' hf => Prod.ext hf hf'


theorem comp₂ {Y₁ Y₂ Z : Type*} {f : X → Y₁} {g : X → Y₂} (hf : IsLocallyConstant f)
    (hg : IsLocallyConstant g) (h : Y₁ → Y₂ → Z) : IsLocallyConstant fun x => h (f x) (g x) :=
  (hf.prod_mk hg).comp fun x : Y₁ × Y₂ => h x.1 x.2


theorem comp_continuous [TopologicalSpace Y] {g : Y → Z} {f : X → Y} (hg : IsLocallyConstant g)
    (hf : Continuous f) : IsLocallyConstant (g ∘ f) := fun s => by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    g : Y → Z
    f : X → Y
    hg : IsLocallyConstant g
    hf : Continuous f
    s : Set Z
    ⊢ IsOpen (Set.preimage (Function.comp g f) s)
  -/
  rw [Set.preimage_comp]
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    g : Y → Z
    f : X → Y
    hg : IsLocallyConstant g
    hf : Continuous f
    s : Set Z
    ⊢ IsOpen (Set.preimage f (Set.preimage g s))
  -/
  exact hf.isOpen_preimage _ (hg _)
  /-
    🎉 no goals
  -/


/-- A locally constant function is constant on any preconnected set. -/
theorem apply_eq_of_isPreconnected {f : X → Y} (hf : IsLocallyConstant f) {s : Set X}
    (hs : IsPreconnected s) {x y : X} (hx : x ∈ s) (hy : y ∈ s) : f x = f y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    hf : IsLocallyConstant f
    s : Set X
    hs : IsPreconnected s
    x y : X
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Eq (f x) (f y)
  -/
  let U := f ⁻¹' {f y}
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    hf : IsLocallyConstant f
    s : Set X
    hs : IsPreconnected s
    x y : X
    hx : Membership.mem s x
    hy : Membership.mem s y
    U : Set X := Set.preimage f (Singleton.singleton (f y))
    ⊢ Eq (f x) (f y)
  -/
  suffices x ∉ Uᶜ from Classical.not_not.1 this
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    hf : IsLocallyConstant f
    s : Set X
    hs : IsPreconnected s
    x y : X
    hx : Membership.mem s x
    hy : Membership.mem s y
    U : Set X := Set.preimage f (Singleton.singleton (f y))
    ⊢ Not (Membership.mem (HasCompl.compl U) x)
  -/
  intro hxV
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    hf : IsLocallyConstant f
    s : Set X
    hs : IsPreconnected s
    x y : X
    hx : Membership.mem s x
    hy : Membership.mem s y
    U : Set X := Set.preimage f (Singleton.singleton (f y))
    hxV : Membership.mem (HasCompl.compl U) x
    ⊢ False
  -/
  specialize hs U Uᶜ (hf {f y}) (hf {f y}ᶜ) _ ⟨y, ⟨hy, rfl⟩⟩ ⟨x, ⟨hx, hxV⟩⟩
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      hf : IsLocallyConstant f
      s : Set X
      hs : IsPreconnected s
      x y : X
      hx : Membership.mem s x
      hy : Membership.mem s y
      U : Set X := Set.preimage f (Singleton.singleton (f y))
      hxV : Membership.mem (HasCompl.compl U) x
      ⊢ HasSubset.Subset s (Union.union U (HasCompl.compl U))
    -/
  · simp only [union_compl_self, subset_univ]
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      hf : IsLocallyConstant f
      s : Set X
      x y : X
      hx : Membership.mem s x
      hy : Membership.mem s y
      U : Set X := Set.preimage f (Singleton.singleton (f y))
      hxV : Membership.mem (HasCompl.compl U) x
      hs : (Inter.inter s (Inter.inter U (HasCompl.compl U))).Nonempty
      ⊢ False
    -/
  · simp only [inter_empty, Set.not_nonempty_empty, inter_compl_self] at hs
    /-
      🎉 no goals
    -/


theorem apply_eq_of_preconnectedSpace [PreconnectedSpace X] {f : X → Y} (hf : IsLocallyConstant f)
    (x y : X) : f x = f y :=
  hf.apply_eq_of_isPreconnected isPreconnected_univ trivial trivial


theorem eq_const [PreconnectedSpace X] {f : X → Y} (hf : IsLocallyConstant f) (x : X) :
    f = Function.const X (f x) :=
  funext fun y => hf.apply_eq_of_preconnectedSpace y x


theorem exists_eq_const [PreconnectedSpace X] [Nonempty Y] {f : X → Y} (hf : IsLocallyConstant f) :
    ∃ y, f = Function.const X y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : PreconnectedSpace X
    inst✝ : Nonempty Y
    f : X → Y
    hf : IsLocallyConstant f
    ⊢ Exists fun y => Eq f (Function.const X y)
  -/
  cases' isEmpty_or_nonempty X with h h
    /-
      case inl
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : PreconnectedSpace X
      inst✝ : Nonempty Y
      f : X → Y
      hf : IsLocallyConstant f
      h : IsEmpty X
      ⊢ Exists fun y => Eq f (Function.const X y)
    -/
  · exact ⟨Classical.arbitrary Y, funext <| h.elim⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : PreconnectedSpace X
      inst✝ : Nonempty Y
      f : X → Y
      hf : IsLocallyConstant f
      h : Nonempty X
      ⊢ Exists fun y => Eq f (Function.const X y)
    -/
  · exact ⟨f (Classical.arbitrary X), hf.eq_const _⟩
    /-
      🎉 no goals
    -/


theorem iff_is_const [PreconnectedSpace X] {f : X → Y} : IsLocallyConstant f ↔ ∀ x y, f x = f y :=
  ⟨fun h _ _ => h.apply_eq_of_isPreconnected isPreconnected_univ trivial trivial, of_constant _⟩


theorem range_finite [CompactSpace X] {f : X → Y} (hf : IsLocallyConstant f) :
    (Set.range f).Finite := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    f : X → Y
    hf : IsLocallyConstant f
    ⊢ (Set.range f).Finite
  -/
  letI : TopologicalSpace Y := ⊥; haveI := discreteTopology_bot Y
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : CompactSpace X
    f : X → Y
    hf : IsLocallyConstant f
    this✝ : TopologicalSpace Y := Bot.bot
    this : DiscreteTopology Y
    ⊢ (Set.range f).Finite
  -/
  exact (isCompact_range hf.continuous).finite_of_discrete
  /-
    🎉 no goals
  -/


@[to_additive]
theorem one [One Y] : IsLocallyConstant (1 : X → Y) := IsLocallyConstant.const 1


@[to_additive]
theorem inv [Inv Y] ⦃f : X → Y⦄ (hf : IsLocallyConstant f) : IsLocallyConstant f⁻¹ :=
  hf.comp fun x => x⁻¹


@[to_additive]
theorem mul [Mul Y] ⦃f g : X → Y⦄ (hf : IsLocallyConstant f) (hg : IsLocallyConstant g) :
    IsLocallyConstant (f * g) :=
  hf.comp₂ hg (· * ·)


@[to_additive]
theorem div [Div Y] ⦃f g : X → Y⦄ (hf : IsLocallyConstant f) (hg : IsLocallyConstant g) :
    IsLocallyConstant (f / g) :=
  hf.comp₂ hg (· / ·)


/-- If a composition of a function `f` followed by an injection `g` is locally
constant, then the locally constant property descends to `f`. -/
theorem desc {α β : Type*} (f : X → α) (g : α → β) (h : IsLocallyConstant (g ∘ f))
    (inj : Function.Injective g) : IsLocallyConstant f := fun s => by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    α : Type u_5
    β : Type u_6
    f : X → α
    g : α → β
    h : IsLocallyConstant (Function.comp g f)
    inj : Function.Injective g
    s : Set α
    ⊢ IsOpen (Set.preimage f s)
  -/
  rw [← preimage_image_eq s inj, preimage_preimage]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    α : Type u_5
    β : Type u_6
    f : X → α
    g : α → β
    h : IsLocallyConstant (Function.comp g f)
    inj : Function.Injective g
    s : Set α
    ⊢ IsOpen (Set.preimage (fun x => g (f x)) (Set.image g s))
  -/
  exact h (g '' s)
  /-
    🎉 no goals
  -/


theorem of_constant_on_connected_components [LocallyConnectedSpace X] {f : X → Y}
    (h : ∀ x, ∀ y ∈ connectedComponent x, f y = f x) : IsLocallyConstant f :=
  (iff_exists_open _).2 fun x =>
    ⟨connectedComponent x, isOpen_connectedComponent, mem_connectedComponent, h x⟩


theorem of_constant_on_connected_clopens [LocallyConnectedSpace X] {f : X → Y}
    (h : ∀ U : Set X, IsConnected U → IsClopen U → ∀ x ∈ U, ∀ y ∈ U, f y = f x) :
    IsLocallyConstant f :=
  of_constant_on_connected_components fun x =>
    h (connectedComponent x) isConnected_connectedComponent isClopen_connectedComponent x
      mem_connectedComponent


theorem of_constant_on_preconnected_clopens [LocallyConnectedSpace X] {f : X → Y}
    (h : ∀ U : Set X, IsPreconnected U → IsClopen U → ∀ x ∈ U, ∀ y ∈ U, f y = f x) :
    IsLocallyConstant f :=
  of_constant_on_connected_clopens fun U hU ↦ h U hU.isPreconnected


/-- A (bundled) locally constant function from a topological space `X` to a type `Y`. -/
structure LocallyConstant (X Y : Type*) [TopologicalSpace X] where
  /-- The underlying function. -/
  protected toFun : X → Y
  /-- The map is locally constant. -/
  protected isLocallyConstant : IsLocallyConstant toFun


instance [Inhabited Y] : Inhabited (LocallyConstant X Y) :=
  ⟨⟨_, IsLocallyConstant.const default⟩⟩


instance : FunLike (LocallyConstant X Y) X Y where
  coe := LocallyConstant.toFun
                       /-
                         X : Type u_1
                         Y : Type u_2
                         Z : Type u_3
                         α : Type u_4
                         inst✝ : TopologicalSpace X
                         ⊢ Function.Injective LocallyConstant.toFun
                       -/
  coe_injective' := by rintro ⟨_, _⟩ ⟨_, _⟩ _; congr
                                               /-
                                                 🎉 no goals
                                               -/


/-- See Note [custom simps projections]. -/
def Simps.apply (f : LocallyConstant X Y) : X → Y := f


@[simp]
theorem toFun_eq_coe (f : LocallyConstant X Y) : f.toFun = f :=
  rfl


@[simp]
theorem coe_mk (f : X → Y) (h) : ⇑(⟨f, h⟩ : LocallyConstant X Y) = f :=
  rfl


protected theorem congr_fun {f g : LocallyConstant X Y} (h : f = g) (x : X) : f x = g x :=
  DFunLike.congr_fun h x


protected theorem congr_arg (f : LocallyConstant X Y) {x y : X} (h : x = y) : f x = f y :=
  DFunLike.congr_arg f h


theorem coe_injective : @Function.Injective (LocallyConstant X Y) (X → Y) (↑) := fun _ _ =>
  DFunLike.ext'


@[norm_cast]
theorem coe_inj {f g : LocallyConstant X Y} : (f : X → Y) = g ↔ f = g :=
  coe_injective.eq_iff


@[ext]
theorem ext ⦃f g : LocallyConstant X Y⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


protected theorem continuous : Continuous f :=
  f.isLocallyConstant.continuous


/-- We can turn a locally-constant function into a bundled `ContinuousMap`. -/
@[coe] def toContinuousMap : C(X, Y) :=
  ⟨f, f.continuous⟩


/-- As a shorthand, `LocallyConstant.toContinuousMap` is available as a coercion -/
instance : Coe (LocallyConstant X Y) C(X, Y) := ⟨toContinuousMap⟩

-- Porting note: became a syntactic `rfl`


@[simp] theorem coe_continuousMap : ((f : C(X, Y)) : X → Y) = (f : X → Y) := rfl


theorem toContinuousMap_injective :
    Function.Injective (toContinuousMap : LocallyConstant X Y → C(X, Y)) := fun _ _ h =>
  ext (ContinuousMap.congr_fun h)


/-- The constant locally constant function on `X` with value `y : Y`. -/
def const (X : Type*) {Y : Type*} [TopologicalSpace X] (y : Y) : LocallyConstant X Y :=
  ⟨Function.const X y, IsLocallyConstant.const _⟩


@[simp]
theorem coe_const (y : Y) : (const X y : X → Y) = Function.const X y :=
  rfl


/-- The locally constant function to `Fin 2` associated to a clopen set. -/
def ofIsClopen {X : Type*} [TopologicalSpace X] {U : Set X} [∀ x, Decidable (x ∈ U)]
    (hU : IsClopen U) : LocallyConstant X (Fin 2) where
  toFun x := if x ∈ U then 0 else 1
  isLocallyConstant := by
    /-
      X✝ : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝² : TopologicalSpace X✝
      X : Type u_5
      inst✝¹ : TopologicalSpace X
      U : Set X
      inst✝ : (x : X) → Decidable (Membership.mem U x)
      hU : IsClopen U
      ⊢ IsLocallyConstant fun x => ite (Membership.mem U x) 0 1
    -/
    refine IsLocallyConstant.iff_isOpen_fiber.2 <| Fin.forall_fin_two.2 ⟨?_, ?_⟩
      /-
        case refine_1
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        ⊢ IsOpen (Set.preimage (fun x => ite (Membership.mem U x) 0 1) (Singleton.sing …
      -/
    · convert hU.2 using 1
      /-
        case h.e'_3
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        ⊢ Eq (Set.preimage (fun x => ite (Membership.mem U x) 0 1) (Singleton.singleto …
      -/
      ext
      simp only [mem_singleton_iff, Fin.one_eq_zero_iff, mem_preimage, ite_eq_left_iff,
        Nat.succ_succ_ne_one]
      /-
        case h.e'_3.h
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        x✝ : X
        ⊢ Iff (Not (Membership.mem U x✝) → False) (Membership.mem U x✝)
      -/
      tauto
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        ⊢ IsOpen (Set.preimage (fun x => ite (Membership.mem U x) 0 1) (Singleton.sing …
      -/
    · rw [← isClosed_compl_iff]
      /-
        case refine_2
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        ⊢ IsClosed (HasCompl.compl (Set.preimage (fun x => ite (Membership.mem U x) 0  …
      -/
      convert hU.1
      /-
        case h.e'_3
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        ⊢ Eq (HasCompl.compl (Set.preimage (fun x => ite (Membership.mem U x) 0 1) (Si …
      -/
      ext
      /-
        case h.e'_3.h
        X✝ : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝² : TopologicalSpace X✝
        X : Type u_5
        inst✝¹ : TopologicalSpace X
        U : Set X
        inst✝ : (x : X) → Decidable (Membership.mem U x)
        hU : IsClopen U
        x✝ : X
        ⊢ Iff (Membership.mem (HasCompl.compl (Set.preimage (fun x => ite (Membership. …
      -/
      simp
      /-
        🎉 no goals
      -/


@[simp]
theorem ofIsClopen_fiber_zero {X : Type*} [TopologicalSpace X] {U : Set X} [∀ x, Decidable (x ∈ U)]
    (hU : IsClopen U) : ofIsClopen hU ⁻¹' ({0} : Set (Fin 2)) = U := by
  /-
    X : Type u_5
    inst✝¹ : TopologicalSpace X
    U : Set X
    inst✝ : (x : X) → Decidable (Membership.mem U x)
    hU : IsClopen U
    ⊢ Eq (Set.preimage (⇑(LocallyConstant.ofIsClopen hU)) (Singleton.singleton 0)) U
  -/
  ext
  simp only [ofIsClopen, mem_singleton_iff, Fin.one_eq_zero_iff, coe_mk, mem_preimage,
    ite_eq_left_iff, Nat.succ_succ_ne_one]
  /-
    case h
    X : Type u_5
    inst✝¹ : TopologicalSpace X
    U : Set X
    inst✝ : (x : X) → Decidable (Membership.mem U x)
    hU : IsClopen U
    x✝ : X
    ⊢ Iff (Not (Membership.mem U x✝) → False) (Membership.mem U x✝)
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
theorem ofIsClopen_fiber_one {X : Type*} [TopologicalSpace X] {U : Set X} [∀ x, Decidable (x ∈ U)]
    (hU : IsClopen U) : ofIsClopen hU ⁻¹' ({1} : Set (Fin 2)) = Uᶜ := by
  /-
    X : Type u_5
    inst✝¹ : TopologicalSpace X
    U : Set X
    inst✝ : (x : X) → Decidable (Membership.mem U x)
    hU : IsClopen U
    ⊢ Eq (Set.preimage (⇑(LocallyConstant.ofIsClopen hU)) (Singleton.singleton 1)) …
  -/
  ext
  simp only [ofIsClopen, mem_singleton_iff, coe_mk, Fin.zero_eq_one_iff, mem_preimage,
    ite_eq_right_iff, mem_compl_iff, Nat.succ_succ_ne_one]


theorem locallyConstant_eq_of_fiber_zero_eq {X : Type*} [TopologicalSpace X]
    (f g : LocallyConstant X (Fin 2)) (h : f ⁻¹' ({0} : Set (Fin 2)) = g ⁻¹' {0}) : f = g := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    f g : LocallyConstant X (Fin 2)
    h : Eq (Set.preimage (⇑f) (Singleton.singleton 0)) (Set.preimage (⇑g) (Singlet …
    ⊢ Eq f g
  -/
  simp only [Set.ext_iff, mem_singleton_iff, mem_preimage] at h
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    f g : LocallyConstant X (Fin 2)
    h : ∀ (x : X), Iff (Eq (f x) 0) (Eq (g x) 0)
    ⊢ Eq f g
  -/
  ext1 x
  /-
    case h
    X : Type u_5
    inst✝ : TopologicalSpace X
    f g : LocallyConstant X (Fin 2)
    h : ∀ (x : X), Iff (Eq (f x) 0) (Eq (g x) 0)
    x : X
    ⊢ Eq (f x) (g x)
  -/
  exact Fin.fin_two_eq_of_eq_zero_iff (h x)
  /-
    🎉 no goals
  -/


theorem range_finite [CompactSpace X] (f : LocallyConstant X Y) : (Set.range f).Finite :=
  f.isLocallyConstant.range_finite


theorem apply_eq_of_isPreconnected (f : LocallyConstant X Y) {s : Set X} (hs : IsPreconnected s)
    {x y : X} (hx : x ∈ s) (hy : y ∈ s) : f x = f y :=
  f.isLocallyConstant.apply_eq_of_isPreconnected hs hx hy


theorem apply_eq_of_preconnectedSpace [PreconnectedSpace X] (f : LocallyConstant X Y) (x y : X) :
    f x = f y :=
  f.isLocallyConstant.apply_eq_of_isPreconnected isPreconnected_univ trivial trivial


theorem eq_const [PreconnectedSpace X] (f : LocallyConstant X Y) (x : X) : f = const X (f x) :=
  ext fun _ => apply_eq_of_preconnectedSpace f _ _


theorem exists_eq_const [PreconnectedSpace X] [Nonempty Y] (f : LocallyConstant X Y) :
    ∃ y, f = const X y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : PreconnectedSpace X
    inst✝ : Nonempty Y
    f : LocallyConstant X Y
    ⊢ Exists fun y => Eq f (LocallyConstant.const X y)
  -/
  rcases Classical.em (Nonempty X) with (⟨⟨x⟩⟩ | hX)
    /-
      case inl.intro
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : PreconnectedSpace X
      inst✝ : Nonempty Y
      f : LocallyConstant X Y
      x : X
      ⊢ Exists fun y => Eq f (LocallyConstant.const X y)
    -/
  · exact ⟨f x, f.eq_const x⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : PreconnectedSpace X
      inst✝ : Nonempty Y
      f : LocallyConstant X Y
      hX : Not (Nonempty X)
      ⊢ Exists fun y => Eq f (LocallyConstant.const X y)
    -/
  · exact ⟨Classical.arbitrary Y, ext fun x => (hX ⟨x⟩).elim⟩
    /-
      🎉 no goals
    -/


/-- Push forward of locally constant maps under any map, by post-composition. -/
def map (f : Y → Z) (g : LocallyConstant X Y) : LocallyConstant X Z :=
  ⟨f ∘ g, g.isLocallyConstant.comp f⟩


@[simp]
theorem map_apply (f : Y → Z) (g : LocallyConstant X Y) : ⇑(map f g) = f ∘ g :=
  rfl


@[simp]
theorem map_id : @map X Y Y _ id = id := rfl


@[simp]
theorem map_comp {Y₁ Y₂ Y₃ : Type*} (g : Y₂ → Y₃) (f : Y₁ → Y₂) :
    @map X _ _ _ g ∘ map f = map (g ∘ f) := rfl


/-- Given a locally constant function to `α → β`, construct a family of locally constant
functions with values in β indexed by α. -/
def flip {X α β : Type*} [TopologicalSpace X] (f : LocallyConstant X (α → β)) (a : α) :
    LocallyConstant X β :=
  f.map fun f => f a


/-- If α is finite, this constructs a locally constant function to `α → β` given a
family of locally constant functions with values in β indexed by α. -/
def unflip {X α β : Type*} [Finite α] [TopologicalSpace X] (f : α → LocallyConstant X β) :
    LocallyConstant X (α → β) where
  toFun x a := f a x
  isLocallyConstant := IsLocallyConstant.iff_isOpen_fiber.2 fun g => by
    have : (fun (x : X) (a : α) => f a x) ⁻¹' {g} = ⋂ a : α, f a ⁻¹' {g a} := by
      ext; simp [funext_iff]
    /-
      X✝ : Type u_1
      Y : Type u_2
      Z : Type u_3
      α✝ : Type u_4
      inst✝² : TopologicalSpace X✝
      X : Type u_5
      α : Type u_6
      β : Type u_7
      inst✝¹ : Finite α
      inst✝ : TopologicalSpace X
      f : α → LocallyConstant X β
      g : α → β
      this : Eq (Set.preimage (fun x a => (f a) x) (Singleton.singleton g)) (Set.iIn …
      ⊢ IsOpen (Set.preimage (fun x a => (f a) x) (Singleton.singleton g))
    -/
    rw [this]
    /-
      X✝ : Type u_1
      Y : Type u_2
      Z : Type u_3
      α✝ : Type u_4
      inst✝² : TopologicalSpace X✝
      X : Type u_5
      α : Type u_6
      β : Type u_7
      inst✝¹ : Finite α
      inst✝ : TopologicalSpace X
      f : α → LocallyConstant X β
      g : α → β
      this : Eq (Set.preimage (fun x a => (f a) x) (Singleton.singleton g)) (Set.iIn …
      ⊢ IsOpen (Set.iInter fun a => Set.preimage (⇑(f a)) (Singleton.singleton (g a)))
    -/
    exact isOpen_iInter_of_finite fun a => (f a).isLocallyConstant _
    /-
      🎉 no goals
    -/


@[simp]
theorem unflip_flip {X α β : Type*} [Finite α] [TopologicalSpace X]
    (f : LocallyConstant X (α → β)) : unflip f.flip = f := rfl


@[simp]
theorem flip_unflip {X α β : Type*} [Finite α] [TopologicalSpace X]
    (f : α → LocallyConstant X β) : (unflip f).flip = f := rfl


/-- Pull back of locally constant maps under a continuous map, by pre-composition. -/
def comap (f : C(X, Y)) (g : LocallyConstant Y Z) : LocallyConstant X Z :=
  ⟨g ∘ f, g.isLocallyConstant.comp_continuous f.continuous⟩


@[simp]
theorem coe_comap (f : C(X, Y)) (g : LocallyConstant Y Z) :
    (comap f g) = g ∘ f := rfl


theorem coe_comap_apply (f : C(X, Y)) (g : LocallyConstant Y Z) (x : X) :
    comap f g x = g (f x) := rfl


@[simp]
theorem comap_id : comap (@ContinuousMap.id X _) = @id (LocallyConstant X Z) := rfl


theorem comap_comp {W : Type*} [TopologicalSpace W] (f : C(W, X)) (g : C(X, Y)) :
    comap (Z := Z) (g.comp f) = comap f ∘ comap g := rfl


theorem comap_comap {W : Type*} [TopologicalSpace W] (f : C(W, X)) (g : C(X, Y))
    (x : LocallyConstant Y Z) : comap f (comap g x) = comap (g.comp f) x := rfl


theorem comap_const (f : C(X, Y)) (y : Y) (h : ∀ x, f x = y) :
    (comap f : LocallyConstant Y Z → LocallyConstant X Z) = fun g => const X (g y) := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    y : Y
    h : ∀ (x : X), Eq (f x) y
    ⊢ Eq (LocallyConstant.comap f) fun g => LocallyConstant.const X (g y)
  -/
  ext; simp [h]
       /-
         🎉 no goals
       -/


lemma comap_injective (f : C(X, Y)) (hfs : f.1.Surjective) :
    (comap (Z := Z) f).Injective := by
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    hfs : Function.Surjective f.toFun
    ⊢ Function.Injective (LocallyConstant.comap f)
  -/
  intro a b h
  /-
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    hfs : Function.Surjective f.toFun
    a b : LocallyConstant Y Z
    h : Eq (LocallyConstant.comap f a) (LocallyConstant.comap f b)
    ⊢ Eq a b
  -/
  ext y
  /-
    case h
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    hfs : Function.Surjective f.toFun
    a b : LocallyConstant Y Z
    h : Eq (LocallyConstant.comap f a) (LocallyConstant.comap f b)
    y : Y
    ⊢ Eq (a y) (b y)
  -/
  obtain ⟨x, hx⟩ := hfs y
  /-
    case h.intro
    X : Type u_1
    Y : Type u_2
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    hfs : Function.Surjective f.toFun
    a b : LocallyConstant Y Z
    h : Eq (LocallyConstant.comap f a) (LocallyConstant.comap f b)
    y : Y
    x : X
    hx : Eq (f.toFun x) y
    ⊢ Eq (a y) (b y)
  -/
  simpa [← hx] using LocallyConstant.congr_fun h x
  /-
    🎉 no goals
  -/


/-- If a locally constant function factors through an injection, then it factors through a locally
constant function. -/
def desc {X α β : Type*} [TopologicalSpace X] {g : α → β} (f : X → α) (h : LocallyConstant X β)
    (cond : g ∘ f = h) (inj : Function.Injective g) : LocallyConstant X α where
  toFun := f
  isLocallyConstant := IsLocallyConstant.desc _ g (cond.symm ▸ h.isLocallyConstant) inj


@[simp]
theorem coe_desc {X α β : Type*} [TopologicalSpace X] (f : X → α) (g : α → β)
    (h : LocallyConstant X β) (cond : g ∘ f = h) (inj : Function.Injective g) :
    ⇑(desc f h cond inj) = f :=
  rfl


/-- Given a clopen set `U` and a locally constant function `f`, `LocallyConstant.mulIndicator`
  returns the locally constant function that is `f` on `U` and `1` otherwise. -/
@[to_additive (attr := simps) "Given a clopen set `U` and a locally constant function `f`,
  `LocallyConstant.indicator` returns the locally constant function that is `f` on `U` and `0`
  otherwise. "]
noncomputable def mulIndicator (hU : IsClopen U) : LocallyConstant X R where
  toFun := Set.mulIndicator U f
  isLocallyConstant := fun s => by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      R : Type u_5
      inst✝ : One R
      U : Set X
      f : LocallyConstant X R
      hU : IsClopen U
      s : Set R
      ⊢ IsOpen (Set.preimage (U.mulIndicator ⇑f) s)
    -/
    rw [mulIndicator_preimage, Set.ite, Set.diff_eq]
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      R : Type u_5
      inst✝ : One R
      U : Set X
      f : LocallyConstant X R
      hU : IsClopen U
      s : Set R
      ⊢ IsOpen (Union.union (Inter.inter (Set.preimage (⇑f) s) U) (Inter.inter (Set. …
    -/
    exact ((f.2 s).inter hU.isOpen).union ((IsLocallyConstant.const 1 s).inter hU.compl.isOpen)
    /-
      🎉 no goals
    -/


open Classical in
@[to_additive]
theorem mulIndicator_apply_eq_if (hU : IsClopen U) :
    mulIndicator f hU a = if a ∈ U then f a else 1 :=
  Set.mulIndicator_apply U f a


@[to_additive]
theorem mulIndicator_of_mem (hU : IsClopen U) (h : a ∈ U) : f.mulIndicator hU a = f a :=
  Set.mulIndicator_of_mem h _


@[to_additive]
theorem mulIndicator_of_not_mem (hU : IsClopen U) (h : a ∉ U) : f.mulIndicator hU a = 1 :=
  Set.mulIndicator_of_not_mem h _


/--
The equivalence between `LocallyConstant X Z` and `LocallyConstant Y Z` given a
homeomorphism `X ≃ₜ Y`
-/
@[simps]
def congrLeft [TopologicalSpace Y] (e : X ≃ₜ Y) : LocallyConstant X Z ≃ LocallyConstant Y Z where
  toFun := comap e.symm
  invFun := comap e
  left_inv := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      e : Homeomorph X Y
      ⊢ Function.LeftInverse (LocallyConstant.comap ↑e) (LocallyConstant.comap ↑e.sy …
    -/
    intro
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      e : Homeomorph X Y
      x✝ : LocallyConstant X Z
      ⊢ Eq (LocallyConstant.comap (↑e) (LocallyConstant.comap (↑e.symm) x✝)) x✝
    -/
    simp [comap_comap]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      e : Homeomorph X Y
      ⊢ Function.RightInverse (LocallyConstant.comap ↑e) (LocallyConstant.comap ↑e.s …
    -/
    intro
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      e : Homeomorph X Y
      x✝ : LocallyConstant Y Z
      ⊢ Eq (LocallyConstant.comap (↑e.symm) (LocallyConstant.comap (↑e) x✝)) x✝
    -/
    simp [comap_comap]
    /-
      🎉 no goals
    -/


/--
The equivalence between `LocallyConstant X Y` and `LocallyConstant X Z` given an
equivalence `Y ≃ Z`
-/
@[simps]
def congrRight (e : Y ≃ Z) : LocallyConstant X Y ≃ LocallyConstant X Z where
  toFun := map e
  invFun := map e.symm
                 /-
                   X : Type u_1
                   Y : Type u_2
                   Z : Type u_3
                   α : Type u_4
                   inst✝ : TopologicalSpace X
                   e : Equiv Y Z
                   ⊢ Function.LeftInverse (LocallyConstant.map ⇑e.symm) (LocallyConstant.map ⇑e)
                 -/
  left_inv := by intro; ext; simp
                             /-
                               🎉 no goals
                             -/
                  /-
                    X : Type u_1
                    Y : Type u_2
                    Z : Type u_3
                    α : Type u_4
                    inst✝ : TopologicalSpace X
                    e : Equiv Y Z
                    ⊢ Function.RightInverse (LocallyConstant.map ⇑e.symm) (LocallyConstant.map ⇑e)
                  -/
  right_inv := by intro; ext; simp
                              /-
                                🎉 no goals
                              -/


variable (X) in
/--
The set of clopen subsets of a topological space is equivalent to the locally constant maps to
a two-element set
-/
def equivClopens [∀ (s : Set X) x, Decidable (x ∈ s)] :
    LocallyConstant X (Fin 2) ≃ TopologicalSpace.Clopens X where
  toFun f := ⟨f ⁻¹' {0}, f.2.isClopen_fiber _⟩
  invFun s := ofIsClopen s.2
                                                            /-
                                                              X : Type u_1
                                                              Y : Type u_2
                                                              Z : Type u_3
                                                              α : Type u_4
                                                              inst✝¹ : TopologicalSpace X
                                                              inst✝ : (s : Set X) → (x : X) → Decidable (Membership.mem s x)
                                                              x✝ : LocallyConstant X (Fin 2)
                                                              ⊢ Eq (Set.preimage (⇑((fun s => LocallyConstant.ofIsClopen ⋯) ((fun f => { car …
                                                            -/
  left_inv _ := locallyConstant_eq_of_fiber_zero_eq _ _ (by simp)
                                                            /-
                                                              🎉 no goals
                                                            -/
                    /-
                      X : Type u_1
                      Y : Type u_2
                      Z : Type u_3
                      α : Type u_4
                      inst✝¹ : TopologicalSpace X
                      inst✝ : (s : Set X) → (x : X) → Decidable (Membership.mem s x)
                      x✝ : TopologicalSpace.Clopens X
                      ⊢ Eq ((fun f => { carrier := Set.preimage (⇑f) (Singleton.singleton 0), isClop …
                    -/
  right_inv _ := by simp
                    /-
                      🎉 no goals
                    -/


/-- Given two closed sets covering a topological space, and locally constant maps on these two sets,
    then if these two locally constant maps agree on the intersection, we get a piecewise defined
    locally constant map on the whole space.

TODO: Generalise this construction to `ContinuousMap`. -/
def piecewise {C₁ C₂ : Set X} (h₁ : IsClosed C₁) (h₂ : IsClosed C₂) (h : C₁ ∪ C₂ = Set.univ)
    (f : LocallyConstant C₁ Z) (g : LocallyConstant C₂ Z)
    (hfg : ∀ (x : X) (hx : x ∈ C₁ ∩ C₂), f ⟨x, hx.1⟩ = g ⟨x, hx.2⟩)
    [DecidablePred (· ∈ C₁)] : LocallyConstant X Z where
  toFun i := if hi : i ∈ C₁ then f ⟨i, hi⟩ else g ⟨i, (Set.compl_subset_iff_union.mpr h) hi⟩
  isLocallyConstant := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      f : LocallyConstant (↑C₁) Z
      g : LocallyConstant (↑C₂) Z
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      ⊢ IsLocallyConstant fun i => dite (Membership.mem C₁ i) (fun hi => f ⟨i, hi⟩)  …
    -/
    let dZ : TopologicalSpace Z := ⊥
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      f : LocallyConstant (↑C₁) Z
      g : LocallyConstant (↑C₂) Z
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      ⊢ IsLocallyConstant fun i => dite (Membership.mem C₁ i) (fun hi => f ⟨i, hi⟩)  …
    -/
    haveI : DiscreteTopology Z := discreteTopology_bot Z
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      f : LocallyConstant (↑C₁) Z
      g : LocallyConstant (↑C₂) Z
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      this : DiscreteTopology Z
      ⊢ IsLocallyConstant fun i => dite (Membership.mem C₁ i) (fun hi => f ⟨i, hi⟩)  …
    -/
    obtain ⟨f, hf⟩ := f
    /-
      case mk
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      g : LocallyConstant (↑C₂) Z
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      this : DiscreteTopology Z
      f : ↑C₁ → Z
      hf : IsLocallyConstant f
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
      ⊢ IsLocallyConstant fun i => dite (Membership.mem C₁ i) (fun hi => { toFun :=  …
    -/
    obtain ⟨g, hg⟩ := g
    /-
      case mk.mk
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      this : DiscreteTopology Z
      f : ↑C₁ → Z
      hf : IsLocallyConstant f
      g : ↑C₂ → Z
      hg : IsLocallyConstant g
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
      ⊢ IsLocallyConstant fun i => dite (Membership.mem C₁ i) (fun hi => { toFun :=  …
    -/
    rw [IsLocallyConstant.iff_continuous] at hf hg ⊢
    /-
      case mk.mk
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      this : DiscreteTopology Z
      f : ↑C₁ → Z
      hf✝ : IsLocallyConstant f
      hf : Continuous f
      g : ↑C₂ → Z
      hg✝ : IsLocallyConstant g
      hg : Continuous g
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
      ⊢ Continuous fun i => dite (Membership.mem C₁ i) (fun hi => { toFun := f, isLo …
    -/
    dsimp only [coe_mk]
    /-
      case mk.mk
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h : Eq (Union.union C₁ C₂) Set.univ
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      this : DiscreteTopology Z
      f : ↑C₁ → Z
      hf✝ : IsLocallyConstant f
      hf : Continuous f
      g : ↑C₂ → Z
      hg✝ : IsLocallyConstant g
      hg : Continuous g
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
      ⊢ Continuous fun i => dite (Membership.mem C₁ i) (fun hi => f ⟨i, hi⟩) fun hi  …
    -/
    rw [Set.union_eq_iUnion] at h
    /-
      case mk.mk
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      α : Type u_4
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h✝ : Eq (Union.union C₁ C₂) Set.univ
      h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      dZ : TopologicalSpace Z := Bot.bot
      this : DiscreteTopology Z
      f : ↑C₁ → Z
      hf✝ : IsLocallyConstant f
      hf : Continuous f
      g : ↑C₂ → Z
      hg✝ : IsLocallyConstant g
      hg : Continuous g
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
      ⊢ Continuous fun i => dite (Membership.mem C₁ i) (fun hi => f ⟨i, hi⟩) fun hi  …
    -/
    refine (locallyFinite_of_finite _).continuous h (fun i ↦ ?_) (fun i ↦ ?_)
      /-
        case mk.mk.refine_1
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝¹ : TopologicalSpace X
        C₁ C₂ : Set X
        h₁ : IsClosed C₁
        h₂ : IsClosed C₂
        h✝ : Eq (Union.union C₁ C₂) Set.univ
        h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
        inst✝ : DecidablePred fun x => Membership.mem C₁ x
        dZ : TopologicalSpace Z := Bot.bot
        this : DiscreteTopology Z
        f : ↑C₁ → Z
        hf✝ : IsLocallyConstant f
        hf : Continuous f
        g : ↑C₂ → Z
        hg✝ : IsLocallyConstant g
        hg : Continuous g
        hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
        i : Bool
        ⊢ IsClosed (cond i C₁ C₂)
      -/
    · cases i <;> [exact h₂; exact h₁]
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.refine_2
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝¹ : TopologicalSpace X
        C₁ C₂ : Set X
        h₁ : IsClosed C₁
        h₂ : IsClosed C₂
        h✝ : Eq (Union.union C₁ C₂) Set.univ
        h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
        inst✝ : DecidablePred fun x => Membership.mem C₁ x
        dZ : TopologicalSpace Z := Bot.bot
        this : DiscreteTopology Z
        f : ↑C₁ → Z
        hf✝ : IsLocallyConstant f
        hf : Continuous f
        g : ↑C₂ → Z
        hg✝ : IsLocallyConstant g
        hg : Continuous g
        hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
        i : Bool
        ⊢ ContinuousOn (fun i => dite (Membership.mem C₁ i) (fun hi => f ⟨i, hi⟩) fun  …
      -/
    · cases i <;> rw [continuousOn_iff_continuous_restrict]
        /-
          case mk.mk.refine_2.false
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₁ C₂ : Set X
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          h✝ : Eq (Union.union C₁ C₂) Set.univ
          h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          dZ : TopologicalSpace Z := Bot.bot
          this : DiscreteTopology Z
          f : ↑C₁ → Z
          hf✝ : IsLocallyConstant f
          hf : Continuous f
          g : ↑C₂ → Z
          hg✝ : IsLocallyConstant g
          hg : Continuous g
          hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
          ⊢ Continuous ((cond Bool.false C₁ C₂).restrict fun i => dite (Membership.mem C …
        -/
      · convert hg
        /-
          case h.e'_5.h
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₁ C₂ : Set X
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          h✝ : Eq (Union.union C₁ C₂) Set.univ
          h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          dZ : TopologicalSpace Z := Bot.bot
          this : DiscreteTopology Z
          f : ↑C₁ → Z
          hf✝ : IsLocallyConstant f
          hf : Continuous f
          g : ↑C₂ → Z
          hg✝ : IsLocallyConstant g
          hg : Continuous g
          hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
          e_1✝ : Eq ↑(cond Bool.false C₁ C₂) ↑C₂
          ⊢ Eq ((cond Bool.false C₁ C₂).restrict fun i => dite (Membership.mem C₁ i) (fu …
        -/
        ext x
        /-
          case h.e'_5.h.h
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₁ C₂ : Set X
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          h✝ : Eq (Union.union C₁ C₂) Set.univ
          h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          dZ : TopologicalSpace Z := Bot.bot
          this : DiscreteTopology Z
          f : ↑C₁ → Z
          hf✝ : IsLocallyConstant f
          hf : Continuous f
          g : ↑C₂ → Z
          hg✝ : IsLocallyConstant g
          hg : Continuous g
          hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
          e_1✝ : Eq ↑(cond Bool.false C₁ C₂) ↑C₂
          x : ↑(cond Bool.false C₁ C₂)
          ⊢ Eq ((cond Bool.false C₁ C₂).restrict (fun i => dite (Membership.mem C₁ i) (f …
        -/
        simp only [cond_false, restrict_apply, Subtype.coe_eta, dite_eq_right_iff]
        /-
          case h.e'_5.h.h
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₁ C₂ : Set X
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          h✝ : Eq (Union.union C₁ C₂) Set.univ
          h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          dZ : TopologicalSpace Z := Bot.bot
          this : DiscreteTopology Z
          f : ↑C₁ → Z
          hf✝ : IsLocallyConstant f
          hf : Continuous f
          g : ↑C₂ → Z
          hg✝ : IsLocallyConstant g
          hg : Continuous g
          hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
          e_1✝ : Eq ↑(cond Bool.false C₁ C₂) ↑C₂
          x : ↑(cond Bool.false C₁ C₂)
          ⊢ ∀ (h : Membership.mem C₁ ↑x), Eq (f ⟨↑x, ⋯⟩) (g x)
        -/
        exact fun hx ↦ hfg x ⟨hx, x.prop⟩
        /-
          🎉 no goals
        -/
        /-
          case mk.mk.refine_2.true
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₁ C₂ : Set X
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          h✝ : Eq (Union.union C₁ C₂) Set.univ
          h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          dZ : TopologicalSpace Z := Bot.bot
          this : DiscreteTopology Z
          f : ↑C₁ → Z
          hf✝ : IsLocallyConstant f
          hf : Continuous f
          g : ↑C₂ → Z
          hg✝ : IsLocallyConstant g
          hg : Continuous g
          hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
          ⊢ Continuous ((cond Bool.true C₁ C₂).restrict fun i => dite (Membership.mem C₁ …
        -/
      · simp only [cond_true, restrict_dite, Subtype.coe_eta]
        /-
          case mk.mk.refine_2.true
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₁ C₂ : Set X
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          h✝ : Eq (Union.union C₁ C₂) Set.univ
          h : Eq (Set.iUnion fun b => cond b C₁ C₂) Set.univ
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          dZ : TopologicalSpace Z := Bot.bot
          this : DiscreteTopology Z
          f : ↑C₁ → Z
          hf✝ : IsLocallyConstant f
          hf : Continuous f
          g : ↑C₂ → Z
          hg✝ : IsLocallyConstant g
          hg : Continuous g
          hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq ({ toFun := f, …
          ⊢ Continuous fun a => f a
        -/
        exact hf
        /-
          🎉 no goals
        -/


@[simp]
lemma piecewise_apply_left {C₁ C₂ : Set X} (h₁ : IsClosed C₁) (h₂ : IsClosed C₂)
    (h : C₁ ∪ C₂ = Set.univ) (f : LocallyConstant C₁ Z) (g : LocallyConstant C₂ Z)
    (hfg : ∀ (x : X) (hx : x ∈ C₁ ∩ C₂), f ⟨x, hx.1⟩ = g ⟨x, hx.2⟩)
    [DecidablePred (· ∈ C₁)] (x : X) (hx : x ∈ C₁) :
    piecewise h₁ h₂ h f g hfg x = f ⟨x, hx⟩ := by
  simp only [piecewise, Set.mem_preimage, continuous_subtype_val.restrictPreimage,
    coe_comap, Function.comp_apply, coe_mk]
  /-
    X : Type u_1
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    C₁ C₂ : Set X
    h₁ : IsClosed C₁
    h₂ : IsClosed C₂
    h : Eq (Union.union C₁ C₂) Set.univ
    f : LocallyConstant (↑C₁) Z
    g : LocallyConstant (↑C₂) Z
    hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
    inst✝ : DecidablePred fun x => Membership.mem C₁ x
    x : X
    hx : Membership.mem C₁ x
    ⊢ Eq (dite (Membership.mem C₁ x) (fun hi => f ⟨x, hi⟩) fun hi => g ⟨x, ⋯⟩) (f  …
  -/
  rw [dif_pos hx]
  /-
    🎉 no goals
  -/


@[simp]
lemma piecewise_apply_right {C₁ C₂ : Set X} (h₁ : IsClosed C₁) (h₂ : IsClosed C₂)
    (h : C₁ ∪ C₂ = Set.univ) (f : LocallyConstant C₁ Z) (g : LocallyConstant C₂ Z)
    (hfg : ∀ (x : X) (hx : x ∈ C₁ ∩ C₂), f ⟨x, hx.1⟩ = g ⟨x, hx.2⟩)
    [DecidablePred (· ∈ C₁)] (x : X) (hx : x ∈ C₂) :
    piecewise h₁ h₂ h f g hfg x = g ⟨x, hx⟩ := by
  simp only [piecewise, Set.mem_preimage, continuous_subtype_val.restrictPreimage,
    coe_comap, Function.comp_apply, coe_mk]
  /-
    X : Type u_1
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    C₁ C₂ : Set X
    h₁ : IsClosed C₁
    h₂ : IsClosed C₂
    h : Eq (Union.union C₁ C₂) Set.univ
    f : LocallyConstant (↑C₁) Z
    g : LocallyConstant (↑C₂) Z
    hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
    inst✝ : DecidablePred fun x => Membership.mem C₁ x
    x : X
    hx : Membership.mem C₂ x
    ⊢ Eq (dite (Membership.mem C₁ x) (fun hi => f ⟨x, hi⟩) fun hi => g ⟨x, ⋯⟩) (g  …
  -/
  split_ifs with h
    /-
      case pos
      X : Type u_1
      Z : Type u_3
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h✝ : Eq (Union.union C₁ C₂) Set.univ
      f : LocallyConstant (↑C₁) Z
      g : LocallyConstant (↑C₂) Z
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      x : X
      hx : Membership.mem C₂ x
      h : Membership.mem C₁ x
      ⊢ Eq (f ⟨x, h⟩) (g ⟨x, hx⟩)
    -/
  · exact hfg x ⟨h, hx⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      Z : Type u_3
      inst✝¹ : TopologicalSpace X
      C₁ C₂ : Set X
      h₁ : IsClosed C₁
      h₂ : IsClosed C₂
      h✝ : Eq (Union.union C₁ C₂) Set.univ
      f : LocallyConstant (↑C₁) Z
      g : LocallyConstant (↑C₂) Z
      hfg : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f ⟨x, ⋯⟩) (g  …
      inst✝ : DecidablePred fun x => Membership.mem C₁ x
      x : X
      hx : Membership.mem C₂ x
      h : Not (Membership.mem C₁ x)
      ⊢ Eq (g ⟨x, ⋯⟩) (g ⟨x, hx⟩)
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- A variant of `LocallyConstant.piecewise` where the two closed sets cover a subset.

TODO: Generalise this construction to `ContinuousMap`. -/
def piecewise' {C₀ C₁ C₂ : Set X} (h₀ : C₀ ⊆ C₁ ∪ C₂) (h₁ : IsClosed C₁)
    (h₂ : IsClosed C₂) (f₁ : LocallyConstant C₁ Z) (f₂ : LocallyConstant C₂ Z)
    [DecidablePred (· ∈ C₁)] (hf : ∀ x (hx : x ∈ C₁ ∩ C₂), f₁ ⟨x, hx.1⟩ = f₂ ⟨x, hx.2⟩) :
    LocallyConstant C₀ Z :=
  letI : ∀ j : C₀, Decidable (j ∈ Subtype.val ⁻¹' C₁) := fun j ↦ decidable_of_iff (↑j ∈ C₁) Iff.rfl
  piecewise (h₁.preimage continuous_subtype_val) (h₂.preimage continuous_subtype_val)
        /-
          X : Type u_1
          Y : Type u_2
          Z : Type u_3
          α : Type u_4
          inst✝¹ : TopologicalSpace X
          C₀ C₁ C₂ : Set X
          h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
          h₁ : IsClosed C₁
          h₂ : IsClosed C₂
          f₁ : LocallyConstant (↑C₁) Z
          f₂ : LocallyConstant (↑C₂) Z
          inst✝ : DecidablePred fun x => Membership.mem C₁ x
          hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
          this : (j : ↑C₀) → Decidable (Membership.mem (Set.preimage Subtype.val C₁) j)  …
          ⊢ Eq (Union.union (Set.preimage Subtype.val C₁) (Set.preimage Subtype.val C₂)) …
        -/
    (by simpa [eq_univ_iff_forall] using h₀)
        /-
          🎉 no goals
        -/
    (f₁.comap ⟨(restrictPreimage C₁ ((↑) : C₀ → X)), continuous_subtype_val.restrictPreimage⟩)
    (f₂.comap ⟨(restrictPreimage C₂ ((↑) : C₀ → X)), continuous_subtype_val.restrictPreimage⟩) <| by
      /-
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝¹ : TopologicalSpace X
        C₀ C₁ C₂ : Set X
        h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
        h₁ : IsClosed C₁
        h₂ : IsClosed C₂
        f₁ : LocallyConstant (↑C₁) Z
        f₂ : LocallyConstant (↑C₂) Z
        inst✝ : DecidablePred fun x => Membership.mem C₁ x
        hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
        this : (j : ↑C₀) → Decidable (Membership.mem (Set.preimage Subtype.val C₁) j)  …
        ⊢ ∀ (x : ↑C₀) (hx : Membership.mem (Inter.inter (Set.preimage Subtype.val C₁)  …
      -/
      rintro ⟨x, hx₀⟩ ⟨hx₁ : x ∈ C₁, hx₂ : x ∈ C₂⟩
      /-
        case mk.intro
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        α : Type u_4
        inst✝¹ : TopologicalSpace X
        C₀ C₁ C₂ : Set X
        h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
        h₁ : IsClosed C₁
        h₂ : IsClosed C₂
        f₁ : LocallyConstant (↑C₁) Z
        f₂ : LocallyConstant (↑C₂) Z
        inst✝ : DecidablePred fun x => Membership.mem C₁ x
        hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
        this : (j : ↑C₀) → Decidable (Membership.mem (Set.preimage Subtype.val C₁) j)  …
        x : X
        hx₀ : Membership.mem C₀ x
        hx₁ : Membership.mem C₁ x
        hx₂ : Membership.mem C₂ x
        ⊢ Eq ((LocallyConstant.comap { toFun := C₁.restrictPreimage Subtype.val, conti …
      -/
      simpa using hf x ⟨hx₁, hx₂⟩
      /-
        🎉 no goals
      -/


@[simp]
lemma piecewise'_apply_left {C₀ C₁ C₂ : Set X} (h₀ : C₀ ⊆ C₁ ∪ C₂) (h₁ : IsClosed C₁)
    (h₂ : IsClosed C₂) (f₁ : LocallyConstant C₁ Z) (f₂ : LocallyConstant C₂ Z)
    [DecidablePred (· ∈ C₁)] (hf : ∀ x (hx : x ∈ C₁ ∩ C₂), f₁ ⟨x, hx.1⟩ = f₂ ⟨x, hx.2⟩)
    (x : C₀) (hx : x.val ∈ C₁) :
    piecewise' h₀ h₁ h₂ f₁ f₂ hf x = f₁ ⟨x.val, hx⟩ := by
  /-
    X : Type u_1
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    C₀ C₁ C₂ : Set X
    h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
    h₁ : IsClosed C₁
    h₂ : IsClosed C₂
    f₁ : LocallyConstant (↑C₁) Z
    f₂ : LocallyConstant (↑C₂) Z
    inst✝ : DecidablePred fun x => Membership.mem C₁ x
    hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
    x : ↑C₀
    hx : Membership.mem C₁ ↑x
    ⊢ Eq ((LocallyConstant.piecewise' h₀ h₁ h₂ f₁ f₂ hf) x) (f₁ ⟨↑x, hx⟩)
  -/
  letI : ∀ j : C₀, Decidable (j ∈ Subtype.val ⁻¹' C₁) := fun j ↦ decidable_of_iff (↑j ∈ C₁) Iff.rfl
  rw [piecewise', piecewise_apply_left (f := (f₁.comap
    ⟨(restrictPreimage C₁ ((↑) : C₀ → X)), continuous_subtype_val.restrictPreimage⟩))
    (hx := hx)]
  /-
    X : Type u_1
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    C₀ C₁ C₂ : Set X
    h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
    h₁ : IsClosed C₁
    h₂ : IsClosed C₂
    f₁ : LocallyConstant (↑C₁) Z
    f₂ : LocallyConstant (↑C₂) Z
    inst✝ : DecidablePred fun x => Membership.mem C₁ x
    hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
    x : ↑C₀
    hx : Membership.mem C₁ ↑x
    this : (j : ↑C₀) → Decidable (Membership.mem (Set.preimage Subtype.val C₁) j)  …
    ⊢ Eq ((LocallyConstant.comap { toFun := C₁.restrictPreimage Subtype.val, conti …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma piecewise'_apply_right {C₀ C₁ C₂ : Set X} (h₀ : C₀ ⊆ C₁ ∪ C₂) (h₁ : IsClosed C₁)
    (h₂ : IsClosed C₂) (f₁ : LocallyConstant C₁ Z) (f₂ : LocallyConstant C₂ Z)
    [DecidablePred (· ∈ C₁)] (hf : ∀ x (hx : x ∈ C₁ ∩ C₂), f₁ ⟨x, hx.1⟩ = f₂ ⟨x, hx.2⟩)
    (x : C₀) (hx : x.val ∈ C₂) :
    piecewise' h₀ h₁ h₂ f₁ f₂ hf x = f₂ ⟨x.val, hx⟩ := by
  /-
    X : Type u_1
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    C₀ C₁ C₂ : Set X
    h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
    h₁ : IsClosed C₁
    h₂ : IsClosed C₂
    f₁ : LocallyConstant (↑C₁) Z
    f₂ : LocallyConstant (↑C₂) Z
    inst✝ : DecidablePred fun x => Membership.mem C₁ x
    hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
    x : ↑C₀
    hx : Membership.mem C₂ ↑x
    ⊢ Eq ((LocallyConstant.piecewise' h₀ h₁ h₂ f₁ f₂ hf) x) (f₂ ⟨↑x, hx⟩)
  -/
  letI : ∀ j : C₀, Decidable (j ∈ Subtype.val ⁻¹' C₁) := fun j ↦ decidable_of_iff (↑j ∈ C₁) Iff.rfl
  rw [piecewise', piecewise_apply_right (f := (f₁.comap
    ⟨(restrictPreimage C₁ ((↑) : C₀ → X)), continuous_subtype_val.restrictPreimage⟩))
    (hx := hx)]
  /-
    X : Type u_1
    Z : Type u_3
    inst✝¹ : TopologicalSpace X
    C₀ C₁ C₂ : Set X
    h₀ : HasSubset.Subset C₀ (Union.union C₁ C₂)
    h₁ : IsClosed C₁
    h₂ : IsClosed C₂
    f₁ : LocallyConstant (↑C₁) Z
    f₂ : LocallyConstant (↑C₂) Z
    inst✝ : DecidablePred fun x => Membership.mem C₁ x
    hf : ∀ (x : X) (hx : Membership.mem (Inter.inter C₁ C₂) x), Eq (f₁ ⟨x, ⋯⟩) (f₂ …
    x : ↑C₀
    hx : Membership.mem C₂ ↑x
    this : (j : ↑C₀) → Decidable (Membership.mem (Set.preimage Subtype.val C₁) j)  …
    ⊢ Eq ((LocallyConstant.comap { toFun := C₂.restrictPreimage Subtype.val, conti …
  -/
  rfl
  /-
    🎉 no goals
  -/


