/-- Let `f : α → α` be a (quasi)ergodic map. Let `g : α → X` is a null-measurable function
from `α` to a nonempty space with a countable family of measurable sets
separating points of a set `s` such that `f x ∈ s` for a.e. `x`.
If `g` that is a.e.-invariant under `f`, then `g` is a.e. constant. -/
theorem QuasiErgodic.ae_eq_const_of_ae_eq_comp_of_ae_range₀ [Nonempty X] [MeasurableSpace X]
    {s : Set X} [MeasurableSpace.CountablySeparated s] {f : α → α} {g : α → X}
    (h : QuasiErgodic f μ) (hs : ∀ᵐ x ∂μ, g x ∈ s) (hgm : NullMeasurable g μ)
    (hg_eq : g ∘ f =ᵐ[μ] g) :
    ∃ c, g =ᵐ[μ] const α c := by
  /-
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Nonempty X
    inst✝¹ : MeasurableSpace X
    s : Set X
    inst✝ : MeasurableSpace.CountablySeparated ↑s
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hs : Filter.Eventually (fun x => Membership.mem s (g x)) (MeasureTheory.ae μ)
    hgm : MeasureTheory.NullMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    ⊢ Exists fun c => (MeasureTheory.ae μ).EventuallyEq g (Function.const α c)
  -/
  refine exists_eventuallyEq_const_of_eventually_mem_of_forall_separating MeasurableSet hs ?_
  /-
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Nonempty X
    inst✝¹ : MeasurableSpace X
    s : Set X
    inst✝ : MeasurableSpace.CountablySeparated ↑s
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hs : Filter.Eventually (fun x => Membership.mem s (g x)) (MeasureTheory.ae μ)
    hgm : MeasureTheory.NullMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    ⊢ ∀ (U : Set X), MeasurableSet U → Or (Filter.Eventually (fun x => Membership. …
  -/
  refine fun U hU ↦ h.ae_mem_or_ae_nmem₀ (s := g ⁻¹' U) (hgm hU) ?_b
  /-
    case _b
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Nonempty X
    inst✝¹ : MeasurableSpace X
    s : Set X
    inst✝ : MeasurableSpace.CountablySeparated ↑s
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hs : Filter.Eventually (fun x => Membership.mem s (g x)) (MeasureTheory.ae μ)
    hgm : MeasureTheory.NullMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    U : Set X
    hU : MeasurableSet U
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Set.preimage f (Set.preimage g U)) (Set.p …
  -/
  refine (hg_eq.mono fun x hx ↦ ?_).set_eq
  /-
    case _b
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : Nonempty X
    inst✝¹ : MeasurableSpace X
    s : Set X
    inst✝ : MeasurableSpace.CountablySeparated ↑s
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hs : Filter.Eventually (fun x => Membership.mem s (g x)) (MeasureTheory.ae μ)
    hgm : MeasureTheory.NullMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    U : Set X
    hU : MeasurableSet U
    x : α
    hx : Eq (Function.comp g f x) (g x)
    ⊢ Iff (Membership.mem (Set.preimage f (Set.preimage g U)) x) (Membership.mem ( …
  -/
  rw [← preimage_comp, mem_preimage, mem_preimage, hx]
  /-
    🎉 no goals
  -/


/-- Let `f : α → α` be a (pre)ergodic map.
Let `g : α → X` be a measurable function from `α` to a nonempty measurable space
with a countable family of measurable sets separating the points of `X`.
If `g` is invariant under `f`, then `g` is a.e. constant. -/
theorem PreErgodic.ae_eq_const_of_ae_eq_comp (h : PreErgodic f μ) (hgm : Measurable g)
    (hg_eq : g ∘ f = g) : ∃ c, g =ᵐ[μ] const α c :=
  exists_eventuallyEq_const_of_forall_separating MeasurableSet fun U hU ↦
                                                      /-
                                                        α : Type u_1
                                                        X : Type u_2
                                                        inst✝³ : MeasurableSpace α
                                                        μ : MeasureTheory.Measure α
                                                        inst✝² : Nonempty X
                                                        inst✝¹ : MeasurableSpace X
                                                        inst✝ : MeasurableSpace.CountablySeparated X
                                                        f : α → α
                                                        g : α → X
                                                        h : PreErgodic f μ
                                                        hgm : Measurable g
                                                        hg_eq : Eq (Function.comp g f) g
                                                        U : Set X
                                                        hU : MeasurableSet U
                                                        ⊢ Eq (Set.preimage f (Set.preimage g U)) (Set.preimage g U)
                                                      -/
    h.ae_mem_or_ae_nmem (s := g ⁻¹' U) (hgm hU) <| by rw [← preimage_comp, hg_eq]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Let `f : α → α` be a quasi ergodic map.
Let `g : α → X` be a null-measurable function from `α` to a nonempty measurable space
with a countable family of measurable sets separating the points of `X`.
If `g` is a.e.-invariant under `f`, then `g` is a.e. constant. -/
theorem QuasiErgodic.ae_eq_const_of_ae_eq_comp₀ (h : QuasiErgodic f μ) (hgm : NullMeasurable g μ)
    (hg_eq : g ∘ f =ᵐ[μ] g) : ∃ c, g =ᵐ[μ] const α c :=
  h.ae_eq_const_of_ae_eq_comp_of_ae_range₀ (s := univ) univ_mem hgm hg_eq


/-- Let `f : α → α` be an ergodic map.
Let `g : α → X` be a null-measurable function from `α` to a nonempty measurable space
with a countable family of measurable sets separating the points of `X`.
If `g` is a.e.-invariant under `f`, then `g` is a.e. constant. -/
theorem Ergodic.ae_eq_const_of_ae_eq_comp₀ (h : Ergodic f μ) (hgm : NullMeasurable g μ)
    (hg_eq : g ∘ f =ᵐ[μ] g) : ∃ c, g =ᵐ[μ] const α c :=
  h.quasiErgodic.ae_eq_const_of_ae_eq_comp₀ hgm hg_eq


/-- Let `f : α → α` be a quasi ergodic map.
Let `g : α → X` be an a.e. strongly measurable function
from `α` to a nonempty metrizable topological space.
If `g` is a.e.-invariant under `f`, then `g` is a.e. constant. -/
theorem ae_eq_const_of_ae_eq_comp_ae {g : α → X} (h : QuasiErgodic f μ)
    (hgm : AEStronglyMeasurable g μ) (hg_eq : g ∘ f =ᵐ[μ] g) : ∃ c, g =ᵐ[μ] const α c := by
  /-
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.MetrizableSpace X
    inst✝ : Nonempty X
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hgm : MeasureTheory.AEStronglyMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    ⊢ Exists fun c => (MeasureTheory.ae μ).EventuallyEq g (Function.const α c)
  -/
  borelize X
  /-
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.MetrizableSpace X
    inst✝ : Nonempty X
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hgm : MeasureTheory.AEStronglyMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ Exists fun c => (MeasureTheory.ae μ).EventuallyEq g (Function.const α c)
  -/
  rcases hgm.isSeparable_ae_range with ⟨t, ht, hgt⟩
  /-
    case intro.intro
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.MetrizableSpace X
    inst✝ : Nonempty X
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hgm : MeasureTheory.AEStronglyMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    t : Set X
    ht : TopologicalSpace.IsSeparable t
    hgt : Filter.Eventually (fun x => Membership.mem t (g x)) (MeasureTheory.ae μ)
    ⊢ Exists fun c => (MeasureTheory.ae μ).EventuallyEq g (Function.const α c)
  -/
  haveI := ht.secondCountableTopology
  /-
    case intro.intro
    α : Type u_1
    X : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace.MetrizableSpace X
    inst✝ : Nonempty X
    f : α → α
    g : α → X
    h : QuasiErgodic f μ
    hgm : MeasureTheory.AEStronglyMeasurable g μ
    hg_eq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    t : Set X
    ht : TopologicalSpace.IsSeparable t
    hgt : Filter.Eventually (fun x => Membership.mem t (g x)) (MeasureTheory.ae μ)
    this : SecondCountableTopology ↑t
    ⊢ Exists fun c => (MeasureTheory.ae μ).EventuallyEq g (Function.const α c)
  -/
  exact h.ae_eq_const_of_ae_eq_comp_of_ae_range₀ hgt hgm.aemeasurable.nullMeasurable hg_eq
  /-
    🎉 no goals
  -/


theorem eq_const_of_compQuasiMeasurePreserving_eq (h : QuasiErgodic f μ) {g : α →ₘ[μ] X}
    (hg_eq : g.compQuasiMeasurePreserving f h.1 = g) : ∃ c, g = .const α c :=
  have : g ∘ f =ᵐ[μ] g := (g.coeFn_compQuasiMeasurePreserving h.1).symm.trans
    (hg_eq.symm ▸ .refl _ _)
  let ⟨c, hc⟩ := h.ae_eq_const_of_ae_eq_comp_ae g.aestronglyMeasurable this
  ⟨c, AEEqFun.ext <| hc.trans (AEEqFun.coeFn_const _ _).symm⟩


/-- Let `f : α → α` be an ergodic map.
Let `g : α → X` be an a.e. strongly measurable function
from `α` to a nonempty metrizable topological space.
If `g` is a.e.-invariant under `f`, then `g` is a.e. constant. -/
theorem ae_eq_const_of_ae_eq_comp_ae {g : α → X} (h : Ergodic f μ) (hgm : AEStronglyMeasurable g μ)
    (hg_eq : g ∘ f =ᵐ[μ] g) : ∃ c, g =ᵐ[μ] const α c :=
  h.quasiErgodic.ae_eq_const_of_ae_eq_comp_ae hgm hg_eq


theorem eq_const_of_compMeasurePreserving_eq (h : Ergodic f μ) {g : α →ₘ[μ] X}
    (hg_eq : g.compMeasurePreserving f h.1 = g) : ∃ c, g = .const α c :=
  h.quasiErgodic.eq_const_of_compQuasiMeasurePreserving_eq hg_eq


