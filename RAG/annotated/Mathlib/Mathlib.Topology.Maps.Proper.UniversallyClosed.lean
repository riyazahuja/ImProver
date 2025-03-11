/-- A map `f : X → Y` is proper if and only if it is continuous and the map
`(Prod.map f id : X × Ultrafilter X → Y × Ultrafilter X)` is closed. This is stronger than
`isProperMap_iff_universally_closed` since it shows that there's only one space to check to get
properness, but in most cases it doesn't matter. -/
theorem isProperMap_iff_isClosedMap_ultrafilter {X : Type u} {Y : Type v} [TopologicalSpace X]
    [TopologicalSpace Y] {f : X → Y} :
    IsProperMap f ↔ Continuous f ∧ IsClosedMap
      (Prod.map f id : X × Ultrafilter X → Y × Ultrafilter X) := by
  -- The proof is basically the same as above.
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (IsProperMap f) (And (Continuous f) (IsClosedMap (Prod.map f id)))
  -/
  constructor <;> intro H
    /-
      case mp
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : IsProperMap f
      ⊢ And (Continuous f) (IsClosedMap (Prod.map f id))
    -/
  · exact ⟨H.continuous, H.universally_closed _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      ⊢ IsProperMap f
    -/
  · rw [isProperMap_iff_ultrafilter]
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      ⊢ And (Continuous f) (∀ ⦃𝒰 : Ultrafilter X⦄ ⦃y : Y⦄, Filter.Tendsto f (↑𝒰) (nh …
    -/
    refine ⟨H.1, fun 𝒰 y hy ↦ ?_⟩
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      y : Y
      hy : Filter.Tendsto f (↑𝒰) (nhds y)
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    let F : Set (X × Ultrafilter X) := closure {xℱ | xℱ.2 = pure xℱ.1}
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      y : Y
      hy : Filter.Tendsto f (↑𝒰) (nhds y)
      F : Set (Prod X (Ultrafilter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pur …
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    have := H.2 F isClosed_closure
    have : (y, 𝒰) ∈ Prod.map f id '' F :=
      this.mem_of_tendsto (hy.prod_mk_nhds (Ultrafilter.tendsto_pure_self 𝒰))
        (Eventually.of_forall fun x ↦ ⟨⟨x, pure x⟩, subset_closure rfl, rfl⟩)
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      𝒰 : Ultrafilter X
      y : Y
      hy : Filter.Tendsto f (↑𝒰) (nhds y)
      F : Set (Prod X (Ultrafilter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pur …
      this✝ : IsClosed (Set.image (Prod.map f id) F)
      this : Membership.mem (Set.image (Prod.map f id) F) { fst := y, snd := 𝒰 }
      ⊢ Exists fun x => And (Eq (f x) y) (LE.le (↑𝒰) (nhds x))
    -/
    rcases this with ⟨⟨x, _⟩, hx, ⟨_, _⟩⟩
    /-
      case mpr.intro.mk.intro.refl
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      F : Set (Prod X (Ultrafilter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pur …
      this : IsClosed (Set.image (Prod.map f id) F)
      x : X
      snd✝ : Ultrafilter X
      hx : Membership.mem F { fst := x, snd := snd✝ }
      hy : Filter.Tendsto f (↑(id snd✝)) (nhds (f x))
      ⊢ Exists fun x_1 => And (Eq (f x_1) (f x)) (LE.le (↑(id snd✝)) (nhds x_1))
    -/
    refine ⟨x, rfl, fun U hU ↦ Ultrafilter.compl_not_mem_iff.mp fun hUc ↦ ?_⟩
    /-
      case mpr.intro.mk.intro.refl
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      F : Set (Prod X (Ultrafilter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pur …
      this : IsClosed (Set.image (Prod.map f id) F)
      x : X
      snd✝ : Ultrafilter X
      hx : Membership.mem F { fst := x, snd := snd✝ }
      hy : Filter.Tendsto f (↑(id snd✝)) (nhds (f x))
      U : Set X
      hU : Membership.mem (nhds x) U
      hUc : Membership.mem (id snd✝) (HasCompl.compl U)
      ⊢ False
    -/
    rw [mem_closure_iff_nhds] at hx
    rcases hx (U ×ˢ {𝒢 | Uᶜ ∈ 𝒢}) (prod_mem_nhds hU ((ultrafilter_isOpen_basic _).mem_nhds hUc))
      with ⟨⟨y, 𝒢⟩, ⟨⟨hy : y ∈ U, hy' : Uᶜ ∈ 𝒢⟩, rfl : 𝒢 = pure y⟩⟩
    /-
      case mpr.intro.mk.intro.refl.intro.mk.intro.intro
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (IsClosedMap (Prod.map f id))
      F : Set (Prod X (Ultrafilter X)) := closure (setOf fun xℱ => Eq xℱ.2 (Pure.pur …
      this : IsClosed (Set.image (Prod.map f id) F)
      x : X
      snd✝ : Ultrafilter X
      hx : ∀ (t : Set (Prod X (Ultrafilter X))), Membership.mem (nhds { fst := x, sn …
      hy✝ : Filter.Tendsto f (↑(id snd✝)) (nhds (f x))
      U : Set X
      hU : Membership.mem (nhds x) U
      hUc : Membership.mem (id snd✝) (HasCompl.compl U)
      y : X
      hy : Membership.mem U y
      hy' : Membership.mem (Pure.pure y) (HasCompl.compl U)
      ⊢ False
    -/
    exact hy' hy
    /-
      🎉 no goals
    -/


/-- A map `f : X → Y` is proper if and only if it is continuous and **universally closed**, in the
sense that for any topological space `Z`, the map `Prod.map f id : X × Z → Y × Z` is closed. Note
that `Z` lives in the same universe as `X` here, but `IsProperMap.universally_closed` does not
have this restriction.

This is taken as the definition of properness in
[N. Bourbaki, *General Topology*][bourbaki1966]. -/
theorem isProperMap_iff_universally_closed {X : Type u} {Y : Type v} [TopologicalSpace X]
    [TopologicalSpace Y] {f : X → Y} :
    IsProperMap f ↔ Continuous f ∧ ∀ (Z : Type u) [TopologicalSpace Z],
      IsClosedMap (Prod.map f id : X × Z → Y × Z) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    ⊢ Iff (IsProperMap f) (And (Continuous f) (∀ (Z : Type u) [inst : TopologicalS …
  -/
  constructor <;> intro H
    /-
      case mp
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : IsProperMap f
      ⊢ And (Continuous f) (∀ (Z : Type u) [inst : TopologicalSpace Z], IsClosedMap  …
    -/
  · exact ⟨H.continuous, fun Z ↦ H.universally_closed _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (∀ (Z : Type u) [inst : TopologicalSpace Z], IsClosedMa …
      ⊢ IsProperMap f
    -/
  · rw [isProperMap_iff_isClosedMap_ultrafilter]
    /-
      case mpr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      f : X → Y
      H : And (Continuous f) (∀ (Z : Type u) [inst : TopologicalSpace Z], IsClosedMa …
      ⊢ And (Continuous f) (IsClosedMap (Prod.map f id))
    -/
    exact ⟨H.1, H.2 _⟩
    /-
      🎉 no goals
    -/

