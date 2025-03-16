/-- Extend a function from a set `A`. The resulting function `g` is such that
at any `x₀`, if `f` converges to some `y` as `x` tends to `x₀` within `A`,
then `g x₀` is defined to be one of these `y`. Else, `g x₀` could be anything. -/
def extendFrom (A : Set X) (f : X → Y) : X → Y :=
  fun x ↦ @limUnder _ _ _ ⟨f x⟩ (𝓝[A] x) f


/-- If `f` converges to some `y` as `x` tends to `x₀` within `A`,
then `f` tends to `extendFrom A f x` as `x` tends to `x₀`. -/
theorem tendsto_extendFrom {A : Set X} {f : X → Y} {x : X} (h : ∃ y, Tendsto f (𝓝[A] x) (𝓝 y)) :
    Tendsto f (𝓝[A] x) (𝓝 <| extendFrom A f x) :=
  tendsto_nhds_limUnder h


theorem extendFrom_eq [T2Space Y] {A : Set X} {f : X → Y} {x : X} {y : Y} (hx : x ∈ closure A)
    (hf : Tendsto f (𝓝[A] x) (𝓝 y)) : extendFrom A f x = y :=
  haveI := mem_closure_iff_nhdsWithin_neBot.mp hx
  tendsto_nhds_unique (tendsto_nhds_limUnder ⟨y, hf⟩) hf


theorem extendFrom_extends [T2Space Y] {f : X → Y} {A : Set X} (hf : ContinuousOn f A) :
    ∀ x ∈ A, extendFrom A f x = f x :=
  fun x x_in ↦ extendFrom_eq (subset_closure x_in) (hf x x_in)


/-- If `f` is a function to a T₃ space `Y` which has a limit within `A` at any
point of a set `B ⊆ closure A`, then `extendFrom A f` is continuous on `B`. -/
theorem continuousOn_extendFrom [RegularSpace Y] {f : X → Y} {A B : Set X} (hB : B ⊆ closure A)
    (hf : ∀ x ∈ B, ∃ y, Tendsto f (𝓝[A] x) (𝓝 y)) : ContinuousOn (extendFrom A f) B := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    ⊢ ContinuousOn (extendFrom A f) B
  -/
  set φ := extendFrom A f
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    ⊢ ContinuousOn φ B
  -/
  intro x x_in
  suffices ∀ V' ∈ 𝓝 (φ x), IsClosed V' → φ ⁻¹' V' ∈ 𝓝[B] x by
    simpa [ContinuousWithinAt, (closed_nhds_basis (φ x)).tendsto_right_iff]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    ⊢ ∀ (V' : Set Y), Membership.mem (nhds (φ x)) V' → IsClosed V' → Membership.me …
  -/
  intro V' V'_in V'_closed
  obtain ⟨V, V_in, V_op, hV⟩ : ∃ V ∈ 𝓝 x, IsOpen V ∧ V ∩ A ⊆ f ⁻¹' V' := by
    have := tendsto_extendFrom (hf x x_in)
    rcases (nhdsWithin_basis_open x A).tendsto_left_iff.mp this V' V'_in with ⟨V, ⟨hxV, V_op⟩, hV⟩
    exact ⟨V, IsOpen.mem_nhds V_op hxV, V_op, hV⟩
  suffices ∀ y ∈ V ∩ B, φ y ∈ V' from
    mem_of_superset (inter_mem_inf V_in <| mem_principal_self B) this
  /-
    case intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    V' : Set Y
    V'_in : Membership.mem (nhds (φ x)) V'
    V'_closed : IsClosed V'
    V : Set X
    V_in : Membership.mem (nhds x) V
    V_op : IsOpen V
    hV : HasSubset.Subset (Inter.inter V A) (Set.preimage f V')
    ⊢ ∀ (y : X), Membership.mem (Inter.inter V B) y → Membership.mem V' (φ y)
  -/
  rintro y ⟨hyV, hyB⟩
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    V' : Set Y
    V'_in : Membership.mem (nhds (φ x)) V'
    V'_closed : IsClosed V'
    V : Set X
    V_in : Membership.mem (nhds x) V
    V_op : IsOpen V
    hV : HasSubset.Subset (Inter.inter V A) (Set.preimage f V')
    y : X
    hyV : Membership.mem V y
    hyB : Membership.mem B y
    ⊢ Membership.mem V' (φ y)
  -/
  haveI := mem_closure_iff_nhdsWithin_neBot.mp (hB hyB)
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    V' : Set Y
    V'_in : Membership.mem (nhds (φ x)) V'
    V'_closed : IsClosed V'
    V : Set X
    V_in : Membership.mem (nhds x) V
    V_op : IsOpen V
    hV : HasSubset.Subset (Inter.inter V A) (Set.preimage f V')
    y : X
    hyV : Membership.mem V y
    hyB : Membership.mem B y
    this : (nhdsWithin y A).NeBot
    ⊢ Membership.mem V' (φ y)
  -/
  have limy : Tendsto f (𝓝[A] y) (𝓝 <| φ y) := tendsto_extendFrom (hf y hyB)
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    V' : Set Y
    V'_in : Membership.mem (nhds (φ x)) V'
    V'_closed : IsClosed V'
    V : Set X
    V_in : Membership.mem (nhds x) V
    V_op : IsOpen V
    hV : HasSubset.Subset (Inter.inter V A) (Set.preimage f V')
    y : X
    hyV : Membership.mem V y
    hyB : Membership.mem B y
    this : (nhdsWithin y A).NeBot
    limy : Filter.Tendsto f (nhdsWithin y A) (nhds (φ y))
    ⊢ Membership.mem V' (φ y)
  -/
  have hVy : V ∈ 𝓝 y := IsOpen.mem_nhds V_op hyV
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    V' : Set Y
    V'_in : Membership.mem (nhds (φ x)) V'
    V'_closed : IsClosed V'
    V : Set X
    V_in : Membership.mem (nhds x) V
    V_op : IsOpen V
    hV : HasSubset.Subset (Inter.inter V A) (Set.preimage f V')
    y : X
    hyV : Membership.mem V y
    hyB : Membership.mem B y
    this : (nhdsWithin y A).NeBot
    limy : Filter.Tendsto f (nhdsWithin y A) (nhds (φ y))
    hVy : Membership.mem (nhds y) V
    ⊢ Membership.mem V' (φ y)
  -/
  have : V ∩ A ∈ 𝓝[A] y := by simpa only [inter_comm] using inter_mem_nhdsWithin A hVy
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A B : Set X
    hB : HasSubset.Subset B (closure A)
    hf : ∀ (x : X), Membership.mem B x → Exists fun y => Filter.Tendsto f (nhdsWit …
    φ : X → Y := extendFrom A f
    x : X
    x_in : Membership.mem B x
    V' : Set Y
    V'_in : Membership.mem (nhds (φ x)) V'
    V'_closed : IsClosed V'
    V : Set X
    V_in : Membership.mem (nhds x) V
    V_op : IsOpen V
    hV : HasSubset.Subset (Inter.inter V A) (Set.preimage f V')
    y : X
    hyV : Membership.mem V y
    hyB : Membership.mem B y
    this✝ : (nhdsWithin y A).NeBot
    limy : Filter.Tendsto f (nhdsWithin y A) (nhds (φ y))
    hVy : Membership.mem (nhds y) V
    this : Membership.mem (nhdsWithin y A) (Inter.inter V A)
    ⊢ Membership.mem V' (φ y)
  -/
  exact V'_closed.mem_of_tendsto limy (mem_of_superset this hV)
  /-
    🎉 no goals
  -/


/-- If a function `f` to a T₃ space `Y` has a limit within a
dense set `A` for any `x`, then `extendFrom A f` is continuous. -/
theorem continuous_extendFrom [RegularSpace Y] {f : X → Y} {A : Set X} (hA : Dense A)
    (hf : ∀ x, ∃ y, Tendsto f (𝓝[A] x) (𝓝 y)) : Continuous (extendFrom A f) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A : Set X
    hA : Dense A
    hf : ∀ (x : X), Exists fun y => Filter.Tendsto f (nhdsWithin x A) (nhds y)
    ⊢ Continuous (extendFrom A f)
  -/
  rw [continuous_iff_continuousOn_univ]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : RegularSpace Y
    f : X → Y
    A : Set X
    hA : Dense A
    hf : ∀ (x : X), Exists fun y => Filter.Tendsto f (nhdsWithin x A) (nhds y)
    ⊢ ContinuousOn (extendFrom A f) Set.univ
  -/
  exact continuousOn_extendFrom (fun x _ ↦ hA x) (by simpa using hf)
  /-
    🎉 no goals
  -/

