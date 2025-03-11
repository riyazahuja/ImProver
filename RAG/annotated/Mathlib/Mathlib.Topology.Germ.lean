/-- The value associated to a germ at a point. This is the common value
shared by all representatives at the given point. -/
def value {X α : Type*} [TopologicalSpace X] {x : X} (φ : Germ (𝓝 x) α) : α :=
                                                  /-
                                                    X✝ : Type u_1
                                                    Y : Type u_2
                                                    Z : Type u_3
                                                    inst✝¹ : TopologicalSpace X✝
                                                    f✝ g✝ : X✝ → Y
                                                    A : Set X✝
                                                    x✝ : X✝
                                                    X : Type u_4
                                                    α : Type u_5
                                                    inst✝ : TopologicalSpace X
                                                    x : X
                                                    φ : (nhds x).Germ α
                                                    f g : X → α
                                                    h : ((nhds x).germSetoid α) f g
                                                    ⊢ Eq ((fun f => f x) f) ((fun f => f x) g)
                                                  -/
  Quotient.liftOn' φ (fun f ↦ f x) fun f g h ↦ by dsimp only; rw [Eventually.self_of_nhds h]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem value_smul {α β : Type*} [SMul α β] (φ : Germ (𝓝 x) α)
    (ψ : Germ (𝓝 x) β) : (φ • ψ).value = φ.value • ψ.value :=
  Germ.inductionOn φ fun _ ↦ Germ.inductionOn ψ fun _ ↦ rfl


/-- The map `Germ (𝓝 x) E → E` into a monoid `E` as a monoid homomorphism -/
@[to_additive "The map `Germ (𝓝 x) E → E` as an additive monoid homomorphism"]
def valueMulHom {X E : Type*} [Monoid E] [TopologicalSpace X] {x : X} : Germ (𝓝 x) E →* E where
  toFun := Filter.Germ.value
  map_one' := rfl
  map_mul' φ ψ := Germ.inductionOn φ fun _ ↦ Germ.inductionOn ψ fun _ ↦ rfl


/-- The map `Germ (𝓝 x) E → E` into a `𝕜`-module `E` as a `𝕜`-linear map -/
def valueₗ {X 𝕜 E : Type*} [Semiring 𝕜] [AddCommMonoid E] [Module 𝕜 E] [TopologicalSpace X]
    {x : X} : Germ (𝓝 x) E →ₗ[𝕜] E where
  __ := Filter.Germ.valueAddHom
  map_smul' := fun _ φ ↦ Germ.inductionOn φ fun _ ↦ rfl


/-- The map `Germ (𝓝 x) E → E` as a ring homomorphism -/
def valueRingHom {X E : Type*} [Semiring E] [TopologicalSpace X] {x : X} : Germ (𝓝 x) E →+* E :=
  { Filter.Germ.valueMulHom, Filter.Germ.valueAddHom with }


/-- The map `Germ (𝓝 x) E → E` as a monotone ring homomorphism -/
def valueOrderRingHom {X E : Type*} [OrderedSemiring E] [TopologicalSpace X] {x : X} :
    Germ (𝓝 x) E →+*o E where
  __ := Filter.Germ.valueRingHom
  monotone' := fun φ ψ ↦
  Germ.inductionOn φ fun _ ↦ Germ.inductionOn ψ fun _ h ↦ h.self_of_nhds


/-- Given a predicate on germs `P : Π x : X, germ (𝓝 x) Y → Prop` and `A : set X`,
build a new predicate on germs `RestrictGermPredicate P A` such that
`(∀ x, RestrictGermPredicate P A x f) ↔ ∀ᶠ x near A, P x f`, see
`forall_restrictGermPredicate_iff` for this equivalence. -/
def RestrictGermPredicate (P : ∀ x : X, Germ (𝓝 x) Y → Prop)
    (A : Set X) : ∀ x : X, Germ (𝓝 x) Y → Prop := fun x φ ↦
  Germ.liftOn φ (fun f ↦ x ∈ A → ∀ᶠ y in 𝓝 x, P y f)
    haveI : ∀ f f' : X → Y, f =ᶠ[𝓝 x] f' → (∀ᶠ y in 𝓝 x, P y f) → ∀ᶠ y in 𝓝 x, P y f' := by
      /-
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        inst✝ : TopologicalSpace X
        f g : X → Y
        A✝ : Set X
        x✝ : X
        P : (x : X) → (nhds x).Germ Y → Prop
        A : Set X
        x : X
        φ : (nhds x).Germ Y
        ⊢ ∀ (f f' : X → Y), (nhds x).EventuallyEq f f' → Filter.Eventually (fun y => P …
      -/
      intro f f' hff' hf
      /-
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        inst✝ : TopologicalSpace X
        f✝ g : X → Y
        A✝ : Set X
        x✝ : X
        P : (x : X) → (nhds x).Germ Y → Prop
        A : Set X
        x : X
        φ : (nhds x).Germ Y
        f f' : X → Y
        hff' : (nhds x).EventuallyEq f f'
        hf : Filter.Eventually (fun y => P y ↑f) (nhds x)
        ⊢ Filter.Eventually (fun y => P y ↑f') (nhds x)
      -/
      apply (hf.and <| Eventually.eventually_nhds hff').mono
      /-
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        inst✝ : TopologicalSpace X
        f✝ g : X → Y
        A✝ : Set X
        x✝ : X
        P : (x : X) → (nhds x).Germ Y → Prop
        A : Set X
        x : X
        φ : (nhds x).Germ Y
        f f' : X → Y
        hff' : (nhds x).EventuallyEq f f'
        hf : Filter.Eventually (fun y => P y ↑f) (nhds x)
        ⊢ ∀ (x : X), And (P x ↑f) (Filter.Eventually (fun x => Eq (f x) (f' x)) (nhds  …
      -/
      rintro y ⟨hy, hy'⟩
      /-
        case intro
        X : Type u_1
        Y : Type u_2
        Z : Type u_3
        inst✝ : TopologicalSpace X
        f✝ g : X → Y
        A✝ : Set X
        x✝ : X
        P : (x : X) → (nhds x).Germ Y → Prop
        A : Set X
        x : X
        φ : (nhds x).Germ Y
        f f' : X → Y
        hff' : (nhds x).EventuallyEq f f'
        hf : Filter.Eventually (fun y => P y ↑f) (nhds x)
        y : X
        hy : P y ↑f
        hy' : Filter.Eventually (fun x => Eq (f x) (f' x)) (nhds y)
        ⊢ P y ↑f'
      -/
      rwa [Germ.coe_eq.mpr (EventuallyEq.symm hy')]
      /-
        🎉 no goals
      -/
    fun f f' hff' ↦ propext <| forall_congr' fun _ ↦ ⟨this f f' hff', this f' f hff'.symm⟩


theorem Filter.Eventually.germ_congr_set
    {P : ∀ x : X, Germ (𝓝 x) Y → Prop} (hf : ∀ᶠ x in 𝓝ˢ A, P x f)
    (h : ∀ᶠ z in 𝓝ˢ A, g z = f z) : ∀ᶠ x in 𝓝ˢ A, P x g := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : Filter.Eventually (fun x => P x ↑f) (nhdsSet A)
    h : Filter.Eventually (fun z => Eq (g z) (f z)) (nhdsSet A)
    ⊢ Filter.Eventually (fun x => P x ↑g) (nhdsSet A)
  -/
  rw [eventually_nhdsSet_iff_forall] at *
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => P y ↑f) (nhds …
    h : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => Eq (g y) (f y) …
    ⊢ ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => P y ↑g) (nhds x)
  -/
  intro x hx
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => P y ↑f) (nhds …
    h : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => Eq (g y) (f y) …
    x : X
    hx : Membership.mem A x
    ⊢ Filter.Eventually (fun y => P y ↑g) (nhds x)
  -/
  apply ((hf x hx).and (h x hx).eventually_nhds).mono
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => P y ↑f) (nhds …
    h : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => Eq (g y) (f y) …
    x : X
    hx : Membership.mem A x
    ⊢ ∀ (x : X), And (P x ↑f) (Filter.Eventually (fun x => Eq (g x) (f x)) (nhds x …
  -/
  intro y hy
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => P y ↑f) (nhds …
    h : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => Eq (g y) (f y) …
    x : X
    hx : Membership.mem A x
    y : X
    hy : And (P y ↑f) (Filter.Eventually (fun x => Eq (g x) (f x)) (nhds y))
    ⊢ P y ↑g
  -/
  convert hy.1 using 1
  /-
    case h.e'_2
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => P y ↑f) (nhds …
    h : ∀ (x : X), Membership.mem A x → Filter.Eventually (fun y => Eq (g y) (f y) …
    x : X
    hx : Membership.mem A x
    y : X
    hy : And (P y ↑f) (Filter.Eventually (fun x => Eq (g x) (f x)) (nhds y))
    ⊢ Eq ↑g ↑f
  -/
  exact Germ.coe_eq.mpr hy.2
  /-
    🎉 no goals
  -/


theorem restrictGermPredicate_congr {P : ∀ x : X, Germ (𝓝 x) Y → Prop}
    (hf : RestrictGermPredicate P A x f) (h : ∀ᶠ z in 𝓝ˢ A, g z = f z) :
    RestrictGermPredicate P A x g := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    x : X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : RestrictGermPredicate P A x ↑f
    h : Filter.Eventually (fun z => Eq (g z) (f z)) (nhdsSet A)
    ⊢ RestrictGermPredicate P A x ↑g
  -/
  intro hx
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    x : X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : RestrictGermPredicate P A x ↑f
    h : Filter.Eventually (fun z => Eq (g z) (f z)) (nhdsSet A)
    hx : Membership.mem A x
    ⊢ Filter.Eventually (fun y => P y ↑g) (nhds x)
  -/
  apply ((hf hx).and <| (eventually_nhdsSet_iff_forall.mp h x hx).eventually_nhds).mono
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    x : X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : RestrictGermPredicate P A x ↑f
    h : Filter.Eventually (fun z => Eq (g z) (f z)) (nhdsSet A)
    hx : Membership.mem A x
    ⊢ ∀ (x : X), And (P x ↑f) (Filter.Eventually (fun x => Eq (g x) (f x)) (nhds x …
  -/
  rintro y ⟨hy, h'y⟩
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f g : X → Y
    A : Set X
    x : X
    P : (x : X) → (nhds x).Germ Y → Prop
    hf : RestrictGermPredicate P A x ↑f
    h : Filter.Eventually (fun z => Eq (g z) (f z)) (nhdsSet A)
    hx : Membership.mem A x
    y : X
    hy : P y ↑f
    h'y : Filter.Eventually (fun x => Eq (g x) (f x)) (nhds y)
    ⊢ P y ↑g
  -/
  rwa [Germ.coe_eq.mpr h'y]
  /-
    🎉 no goals
  -/


theorem forall_restrictGermPredicate_iff {P : ∀ x : X, Germ (𝓝 x) Y → Prop} :
    (∀ x, RestrictGermPredicate P A x f) ↔ ∀ᶠ x in 𝓝ˢ A, P x f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    ⊢ Iff (∀ (x : X), RestrictGermPredicate P A x ↑f) (Filter.Eventually (fun x => …
  -/
  rw [eventually_nhdsSet_iff_forall]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    A : Set X
    P : (x : X) → (nhds x).Germ Y → Prop
    ⊢ Iff (∀ (x : X), RestrictGermPredicate P A x ↑f) (∀ (x : X), Membership.mem A …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem forall_restrictGermPredicate_of_forall
    {P : ∀ x : X, Germ (𝓝 x) Y → Prop} (h : ∀ x, P x f) :
    ∀ x, RestrictGermPredicate P A x f :=
  forall_restrictGermPredicate_iff.mpr (Eventually.of_forall h)

/-- Map the germ of functions `X × Y → Z` at `p = (x,y) ∈ X × Y` to the corresponding germ
  of functions `X → Z` at `x ∈ X` -/
def sliceLeft [TopologicalSpace Y] {p : X × Y} (P : Germ (𝓝 p) Z) : Germ (𝓝 p.1) Z :=
  P.compTendsto (Prod.mk · p.2) (Continuous.Prod.mk_left p.2).continuousAt


@[simp]
theorem sliceLeft_coe [TopologicalSpace Y] {y : Y} (f : X × Y → Z) :
    (↑f : Germ (𝓝 (x, y)) Z).sliceLeft = fun x' ↦ f (x', y) :=
  rfl


/-- Map the germ of functions `X × Y → Z` at `p = (x,y) ∈ X × Y` to the corresponding germ
  of functions `Y → Z` at `y ∈ Y` -/
def sliceRight [TopologicalSpace Y] {p : X × Y} (P : Germ (𝓝 p) Z) : Germ (𝓝 p.2) Z :=
  P.compTendsto (Prod.mk p.1) (Continuous.Prod.mk p.1).continuousAt


@[simp]
theorem sliceRight_coe [TopologicalSpace Y] {y : Y} (f : X × Y → Z) :
    (↑f : Germ (𝓝 (x, y)) Z).sliceRight = fun y' ↦ f (x, y') :=
  rfl


lemma isConstant_comp_subtype {s : Set X} {f : X → Y} {x : s}
    (hf : (f : Germ (𝓝 (x : X)) Y).IsConstant) :
    ((f ∘ Subtype.val : s → Y) : Germ (𝓝 x) Y).IsConstant :=
  isConstant_comp_tendsto hf continuousAt_subtype_val


/-- If the germ of `f` w.r.t. each `𝓝 x` is constant, `f` is locally constant. -/
lemma IsLocallyConstant.of_germ_isConstant (h : ∀ x : X, (f : Germ (𝓝 x) Y).IsConstant) :
    IsLocallyConstant f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    ⊢ IsLocallyConstant f
  -/
  intro s
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    ⊢ IsOpen (Set.preimage f s)
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    ⊢ ∀ (x : X), Membership.mem (Set.preimage f s) x → Membership.mem (nhds x) (Se …
  -/
  intro a ha
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    a : X
    ha : Membership.mem (Set.preimage f s) a
    ⊢ Membership.mem (nhds a) (Set.preimage f s)
  -/
  obtain ⟨b, hb⟩ := h a
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    a : X
    ha : Membership.mem (Set.preimage f s) a
    b : Y
    hb : (nhds a).EventuallyEq f fun x => b
    ⊢ Membership.mem (nhds a) (Set.preimage f s)
  -/
  apply mem_of_superset hb
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    a : X
    ha : Membership.mem (Set.preimage f s) a
    b : Y
    hb : (nhds a).EventuallyEq f fun x => b
    ⊢ HasSubset.Subset (setOf fun x => (fun x => Eq (f x) ((fun x => b) x)) x) (Se …
  -/
  intro x hx
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    a : X
    ha : Membership.mem (Set.preimage f s) a
    b : Y
    hb : (nhds a).EventuallyEq f fun x => b
    x : X
    hx : Membership.mem (setOf fun x => (fun x => Eq (f x) ((fun x => b) x)) x) x
    ⊢ Membership.mem (Set.preimage f s) x
  -/
  have : f x = f a := (mem_of_mem_nhds hb) ▸ hx
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    a : X
    ha : Membership.mem (Set.preimage f s) a
    b : Y
    hb : (nhds a).EventuallyEq f fun x => b
    x : X
    hx : Membership.mem (setOf fun x => (fun x => Eq (f x) ((fun x => b) x)) x) x
    this : Eq (f x) (f a)
    ⊢ Membership.mem (Set.preimage f s) x
  -/
  rw [mem_preimage, this]
  /-
    case intro
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    h : ∀ (x : X), (↑f).IsConstant
    s : Set Y
    a : X
    ha : Membership.mem (Set.preimage f s) a
    b : Y
    hb : (nhds a).EventuallyEq f fun x => b
    x : X
    hx : Membership.mem (setOf fun x => (fun x => Eq (f x) ((fun x => b) x)) x) x
    this : Eq (f x) (f a)
    ⊢ Membership.mem s (f a)
  -/
  exact ha
  /-
    🎉 no goals
  -/


theorem eq_of_germ_isConstant [i : PreconnectedSpace X]
    (h : ∀ x : X, (f : Germ (𝓝 x) Y).IsConstant) (x x' : X) : f x = f x' :=
  (IsLocallyConstant.of_germ_isConstant h).apply_eq_of_isPreconnected
                                          /-
                                            X : Type u_1
                                            Y : Type u_2
                                            inst✝ : TopologicalSpace X
                                            f : X → Y
                                            i : PreconnectedSpace X
                                            h : ∀ (x : X), (↑f).IsConstant
                                            x x' : X
                                            ⊢ Membership.mem Set.univ x
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
    (preconnectedSpace_iff_univ.mp i) (by trivial) (by trivial)
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma eq_of_germ_isConstant_on {s : Set X} (h : ∀ x ∈ s, (f : Germ (𝓝 x) Y).IsConstant)
    (hs : IsPreconnected s) {x' : X} (x_in : x ∈ s) (x'_in : x' ∈ s) : f x = f x' := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    x : X
    s : Set X
    h : ∀ (x : X), Membership.mem s x → (↑f).IsConstant
    hs : IsPreconnected s
    x' : X
    x_in : Membership.mem s x
    x'_in : Membership.mem s x'
    ⊢ Eq (f x) (f x')
  -/
  let i : s → X := fun x ↦ x
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    x : X
    s : Set X
    h : ∀ (x : X), Membership.mem s x → (↑f).IsConstant
    hs : IsPreconnected s
    x' : X
    x_in : Membership.mem s x
    x'_in : Membership.mem s x'
    i : ↑s → X := fun x => ↑x
    ⊢ Eq (f x) (f x')
  -/
  show (f ∘ i) (⟨x, x_in⟩ : s) = (f ∘ i) (⟨x', x'_in⟩ : s)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    x : X
    s : Set X
    h : ∀ (x : X), Membership.mem s x → (↑f).IsConstant
    hs : IsPreconnected s
    x' : X
    x_in : Membership.mem s x
    x'_in : Membership.mem s x'
    i : ↑s → X := fun x => ↑x
    ⊢ Eq (Function.comp f i ⟨x, x_in⟩) (Function.comp f i ⟨x', x'_in⟩)
  -/
  have : PreconnectedSpace s := Subtype.preconnectedSpace hs
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    x : X
    s : Set X
    h : ∀ (x : X), Membership.mem s x → (↑f).IsConstant
    hs : IsPreconnected s
    x' : X
    x_in : Membership.mem s x
    x'_in : Membership.mem s x'
    i : ↑s → X := fun x => ↑x
    this : PreconnectedSpace ↑s
    ⊢ Eq (Function.comp f i ⟨x, x_in⟩) (Function.comp f i ⟨x', x'_in⟩)
  -/
  exact eq_of_germ_isConstant (fun y ↦ Germ.isConstant_comp_subtype (h y y.2)) _ _
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem Germ.coe_prod {α : Type*} (l : Filter α) (R : Type*) [CommMonoid R] {ι} (f : ι → α → R)
    (s : Finset ι) : ((∏ i ∈ s, f i : α → R) : Germ l R) = ∏ i ∈ s, (f i : Germ l R) :=
  map_prod (Germ.coeMulHom l : (α → R) →* Germ l R) f s

