/-- If `X` is a cofiltered limit of profinite sets, then any clopen subset of `X` arises from
a clopen set in one of the terms in the limit.
-/
theorem exists_isClopen_of_cofiltered {U : Set C.pt} (hC : IsLimit C) (hU : IsClopen U) :
    ∃ (j : J) (V : Set (F.obj j)), IsClopen V ∧ U = C.π.app j ⁻¹' V := by
  -- First, we have the topological basis of the cofiltered limit obtained by pulling back
  -- clopen sets from the factors in the limit. By continuity, all such sets are again clopen.
  have hB := TopCat.isTopologicalBasis_cofiltered_limit.{u, v} (F ⋙ Profinite.toTopCat)
      (Profinite.toTopCat.mapCone C) (isLimitOfPreserves _ hC) (fun j => {W | IsClopen W}) ?_
      (fun i => isClopen_univ) (fun i U1 U2 hU1 hU2 => hU1.inter hU2) ?_
  /-
    case refine_3
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    hB : TopologicalSpace.IsTopologicalBasis (setOf fun U => Exists fun j => Exist …
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  rotate_left
    /-
      case refine_1
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      U : Set ↑C.pt.toTop
      hC : CategoryTheory.Limits.IsLimit C
      hU : IsClopen U
      ⊢ ∀ (j : J), TopologicalSpace.IsTopologicalBasis ((fun j => setOf fun W => IsC …
    -/
  · intro i
    /-
      case refine_1
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      U : Set ↑C.pt.toTop
      hC : CategoryTheory.Limits.IsLimit C
      hU : IsClopen U
      i : J
      ⊢ TopologicalSpace.IsTopologicalBasis ((fun j => setOf fun W => IsClopen W) i)
    -/
    change TopologicalSpace.IsTopologicalBasis {W : Set (F.obj i) | IsClopen W}
    /-
      case refine_1
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      U : Set ↑C.pt.toTop
      hC : CategoryTheory.Limits.IsLimit C
      hU : IsClopen U
      i : J
      ⊢ TopologicalSpace.IsTopologicalBasis (setOf fun W => IsClopen W)
    -/
    apply isTopologicalBasis_isClopen
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      U : Set ↑C.pt.toTop
      hC : CategoryTheory.Limits.IsLimit C
      hU : IsClopen U
      ⊢ ∀ (i j : J) (f : Quiver.Hom i j) (V : Set ↑((F.comp Profinite.toTopCat).obj  …
    -/
  · rintro i j f V (hV : IsClopen _)
    exact ⟨hV.1.preimage ((F ⋙ toTopCat).map f).continuous,
      hV.2.preimage ((F ⋙ toTopCat).map f).continuous⟩
    -- Porting note: `<;> continuity` fails
  -- Using this, since `U` is open, we can write `U` as a union of clopen sets all of which
  -- are preimages of clopens from the factors in the limit.
  /-
    case refine_3
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    hB : TopologicalSpace.IsTopologicalBasis (setOf fun U => Exists fun j => Exist …
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  obtain ⟨S, hS, h⟩ := hB.open_eq_sUnion hU.2
  /-
    case refine_3.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    hB : TopologicalSpace.IsTopologicalBasis (setOf fun U => Exists fun j => Exist …
    S : Set (Set ↑(Profinite.toTopCat.mapCone C).pt)
    hS : HasSubset.Subset S (setOf fun U => Exists fun j => Exists fun V => And (M …
    h : Eq U S.sUnion
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  clear hB
  /-
    case refine_3.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    S : Set (Set ↑(Profinite.toTopCat.mapCone C).pt)
    hS : HasSubset.Subset S (setOf fun U => Exists fun j => Exists fun V => And (M …
    h : Eq U S.sUnion
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  let j : S → J := fun s => (hS s.2).choose
  /-
    case refine_3.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    S : Set (Set ↑(Profinite.toTopCat.mapCone C).pt)
    hS : HasSubset.Subset S (setOf fun U => Exists fun j => Exists fun V => And (M …
    h : Eq U S.sUnion
    j : ↑S → J := fun s => Exists.choose ⋯
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  let V : ∀ s : S, Set (F.obj (j s)) := fun s => (hS s.2).choose_spec.choose
  have hV : ∀ s : S, IsClopen (V s) ∧ s.1 = C.π.app (j s) ⁻¹' V s := fun s =>
    (hS s.2).choose_spec.choose_spec

  -- Since `U` is also closed, hence compact, it is covered by finitely many of the
  -- clopens constructed in the previous step.
  have hUo : ∀ (i : ↑S), IsOpen ((fun s ↦ (forget Profinite).map (C.π.app (j s)) ⁻¹' V s) i) := by
    intro s
    exact (hV s).1.2.preimage (C.π.app (j s)).continuous
  have hsU : U ⊆ ⋃ (i : ↑S), (fun s ↦ (forget Profinite).map (C.π.app (j s)) ⁻¹' V s) i := by
    dsimp only
    rw [h]
    rintro x ⟨T, hT, hx⟩
    refine ⟨_, ⟨⟨T, hT⟩, rfl⟩, ?_⟩
    dsimp only [forget_map_eq_coe]
    rwa [← (hV ⟨T, hT⟩).2]
  /-
    case refine_3.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    S : Set (Set ↑(Profinite.toTopCat.mapCone C).pt)
    hS : HasSubset.Subset S (setOf fun U => Exists fun j => Exists fun V => And (M …
    h : Eq U S.sUnion
    j : ↑S → J := fun s => Exists.choose ⋯
    V : (s : ↑S) → Set ↑(F.obj (j s)).toTop := fun s => ⋯.choose
    hV : ∀ (s : ↑S), And (IsClopen (V s)) (Eq (↑s) (Set.preimage (⇑(C.π.app (j s)) …
    hUo : ∀ (i : ↑S), IsOpen ((fun s => Set.preimage ((CategoryTheory.forget Profi …
    hsU : HasSubset.Subset U (Set.iUnion fun i => (fun s => Set.preimage ((Categor …
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  have := hU.1.isCompact.elim_finite_subcover (fun s : S => C.π.app (j s) ⁻¹' V s) hUo hsU
  -- Porting note: same remark as after `hB`
  -- We thus obtain a finite set `G : Finset J` and a clopen set of `F.obj j` for each
  -- `j ∈ G` such that `U` is the union of the preimages of these clopen sets.
  /-
    case refine_3.intro.intro
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    U : Set ↑C.pt.toTop
    hC : CategoryTheory.Limits.IsLimit C
    hU : IsClopen U
    S : Set (Set ↑(Profinite.toTopCat.mapCone C).pt)
    hS : HasSubset.Subset S (setOf fun U => Exists fun j => Exists fun V => And (M …
    h : Eq U S.sUnion
    j : ↑S → J := fun s => Exists.choose ⋯
    V : (s : ↑S) → Set ↑(F.obj (j s)).toTop := fun s => ⋯.choose
    hV : ∀ (s : ↑S), And (IsClopen (V s)) (Eq (↑s) (Set.preimage (⇑(C.π.app (j s)) …
    hUo : ∀ (i : ↑S), IsOpen ((fun s => Set.preimage ((CategoryTheory.forget Profi …
    hsU : HasSubset.Subset U (Set.iUnion fun i => (fun s => Set.preimage ((Categor …
    this : Exists fun t => HasSubset.Subset U (Set.iUnion fun i => Set.iUnion fun  …
    ⊢ Exists fun j => Exists fun V => And (IsClopen V) (Eq U (Set.preimage (⇑(C.π. …
  -/
  obtain ⟨G, hG⟩ := this
  -- Since `J` is cofiltered, we can find a single `j0` dominating all the `j ∈ G`.
  -- Pulling back all of the sets from the previous step to `F.obj j0` and taking a union,
  -- we obtain a clopen set in `F.obj j0` which works.
  classical
  obtain ⟨j0, hj0⟩ := IsCofiltered.inf_objs_exists (G.image j)
  let f : ∀ s ∈ G, j0 ⟶ j s := fun s hs => (hj0 (Finset.mem_image.mpr ⟨s, hs, rfl⟩)).some
  let W : S → Set (F.obj j0) := fun s => if hs : s ∈ G then F.map (f s hs) ⁻¹' V s else Set.univ
  -- Conclude, using the `j0` and the clopen set of `F.obj j0` obtained above.
  refine ⟨j0, ⋃ (s : S) (_ : s ∈ G), W s, ?_, ?_⟩
  · apply isClopen_biUnion_finset
    intro s hs
    dsimp [W]
    rw [dif_pos hs]
    exact ⟨(hV s).1.1.preimage (F.map _).continuous, (hV s).1.2.preimage (F.map _).continuous⟩
  · ext x
    constructor
    · intro hx
      simp_rw [W, Set.preimage_iUnion, Set.mem_iUnion]
      obtain ⟨_, ⟨s, rfl⟩, _, ⟨hs, rfl⟩, hh⟩ := hG hx
      refine ⟨s, hs, ?_⟩
      rwa [dif_pos hs, ← Set.preimage_comp, ← CompHausLike.coe_comp, ← Functor.map_comp, C.w]
    · intro hx
      simp_rw [W, Set.preimage_iUnion, Set.mem_iUnion] at hx
      obtain ⟨s, hs, hx⟩ := hx
      rw [h]
      refine ⟨s.1, s.2, ?_⟩
      rw [(hV s).2]
      rwa [dif_pos hs, ← Set.preimage_comp, ← CompHausLike.coe_comp, ← Functor.map_comp, C.w] at hx


theorem exists_locallyConstant_fin_two (hC : IsLimit C) (f : LocallyConstant C.pt (Fin 2)) :
    ∃ (j : J) (g : LocallyConstant (F.obj j) (Fin 2)), f = g.comap (C.π.app _) := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) (Fin 2)
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  let U := f ⁻¹' {0}
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) (Fin 2)
    U : Set ↑C.pt.toTop := Set.preimage (⇑f) (Singleton.singleton 0)
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  have hU : IsClopen U := f.isLocallyConstant.isClopen_fiber _
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) (Fin 2)
    U : Set ↑C.pt.toTop := Set.preimage (⇑f) (Singleton.singleton 0)
    hU : IsClopen U
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  obtain ⟨j, V, hV, h⟩ := exists_isClopen_of_cofiltered C hC hU
  classical
  use j, LocallyConstant.ofIsClopen hV
  apply LocallyConstant.locallyConstant_eq_of_fiber_zero_eq
  simp only [Fin.isValue, Functor.const_obj_obj, LocallyConstant.coe_comap, Set.preimage_comp,
    LocallyConstant.ofIsClopen_fiber_zero]
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [← h]


open Classical in
theorem exists_locallyConstant_finite_aux {α : Type*} [Finite α] (hC : IsLimit C)
    (f : LocallyConstant C.pt α) : ∃ (j : J) (g : LocallyConstant (F.obj j) (α → Fin 2)),
      (f.map fun a b => if a = b then (0 : Fin 2) else 1) = g.comap (C.π.app _) := by
  /-
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  cases nonempty_fintype α
  /-
    case intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  let ι : α → α → Fin 2 := fun x y => if x = y then 0 else 1
  /-
    case intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  let ff := (f.map ι).flip
  /-
    case intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  have hff := fun a : α => exists_locallyConstant_fin_two _ hC (ff a)
  /-
    case intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    hff : ∀ (a : α), Exists fun j => Exists fun g => Eq (ff a) (LocallyConstant.co …
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  choose j g h using hff
  /-
    case intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  let G : Finset J := Finset.univ.image j
  /-
    case intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  obtain ⟨j0, hj0⟩ := IsCofiltered.inf_objs_exists G
  have hj : ∀ a, j a ∈ (Finset.univ.image j : Finset J) := by
    intro a
    simp only [Finset.mem_image, Finset.mem_univ, true_and, exists_apply_eq_apply]
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  let fs : ∀ a : α, j0 ⟶ j a := fun a => (hj0 (hj a)).some
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  let gg : α → LocallyConstant (F.obj j0) (Fin 2) := fun a => (g a).comap (F.map (fs _))
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  let ggg := LocallyConstant.unflip gg
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    ⊢ Exists fun j => Exists fun g => Eq (LocallyConstant.map (fun a b => ite (Eq  …
  -/
  refine ⟨j0, ggg, ?_⟩
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    ⊢ Eq (LocallyConstant.map (fun a b => ite (Eq a b) 0 1) f) (LocallyConstant.co …
  -/
  have : f.map ι = LocallyConstant.unflip (f.map ι).flip := by simp
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    this : Eq (LocallyConstant.map ι f) (LocallyConstant.unflip (LocallyConstant.m …
    ⊢ Eq (LocallyConstant.map (fun a b => ite (Eq a b) 0 1) f) (LocallyConstant.co …
  -/
  rw [this]; clear this
  have :
    LocallyConstant.comap (C.π.app j0) ggg =
      LocallyConstant.unflip (LocallyConstant.comap (C.π.app j0) ggg).flip := by
    simp
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    this : Eq (LocallyConstant.comap (C.π.app j0) ggg) (LocallyConstant.unflip (Lo …
    ⊢ Eq (LocallyConstant.unflip (LocallyConstant.map ι f).flip) (LocallyConstant. …
  -/
  rw [this]; clear this
  /-
    case intro.intro
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    ⊢ Eq (LocallyConstant.unflip (LocallyConstant.map ι f).flip) (LocallyConstant. …
  -/
  congr 1
  /-
    case intro.intro.e_f
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    ⊢ Eq (LocallyConstant.map ι f).flip (LocallyConstant.comap (C.π.app j0) ggg).f …
  -/
  ext1 a
  /-
    case intro.intro.e_f.h
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    a : α
    ⊢ Eq ((LocallyConstant.map ι f).flip a) ((LocallyConstant.comap (C.π.app j0) g …
  -/
  change ff a = _
  /-
    case intro.intro.e_f.h
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    a : α
    ⊢ Eq (ff a) ((LocallyConstant.comap (C.π.app j0) ggg).flip a)
  -/
  rw [h]
  /-
    case intro.intro.e_f.h
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    a : α
    ⊢ Eq (LocallyConstant.comap (C.π.app (j a)) (g a)) ((LocallyConstant.comap (C. …
  -/
  dsimp
  /-
    case intro.intro.e_f.h
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    a : α
    ⊢ Eq (LocallyConstant.comap (C.π.app (j a)) (g a)) ((LocallyConstant.comap (C. …
  -/
  ext1 x
  /-
    case intro.intro.e_f.h.h
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    a : α
    x : ↑C.pt.toTop
    ⊢ Eq ((LocallyConstant.comap (C.π.app (j a)) (g a)) x) (((LocallyConstant.coma …
  -/
  change _ = (g a) ((C.π.app j0 ≫ F.map (fs a)) x)
  /-
    case intro.intro.e_f.h.h
    J : Type v
    inst✝² : CategoryTheory.SmallCategory J
    inst✝¹ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝ : Finite α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    val✝ : Fintype α
    ι : α → α → Fin 2 := fun x y => ite (Eq x y) 0 1
    ff : α → LocallyConstant (↑C.pt.toTop) (Fin 2) := (LocallyConstant.map ι f).flip
    j : α → J
    g : (a : α) → LocallyConstant (↑(F.obj (j a)).toTop) (Fin 2)
    h : ∀ (a : α), Eq (ff a) (LocallyConstant.comap (C.π.app (j a)) (g a))
    G : Finset J := Finset.image j Finset.univ
    j0 : J
    hj0 : ∀ {X : J}, Membership.mem G X → Nonempty (Quiver.Hom j0 X)
    hj : ∀ (a : α), Membership.mem (Finset.image j Finset.univ) (j a)
    fs : (a : α) → Quiver.Hom j0 (j a) := fun a => ⋯.some
    gg : α → LocallyConstant (↑(F.obj j0).toTop) (Fin 2) := fun a => LocallyConsta …
    ggg : LocallyConstant (↑(F.obj j0).toTop) (α → Fin 2) := LocallyConstant.unfli …
    a : α
    x : ↑C.pt.toTop
    ⊢ Eq ((LocallyConstant.comap (C.π.app (j a)) (g a)) x) ((g a) ((CategoryTheory …
  -/
  rw [C.w]; rfl
            /-
              🎉 no goals
            -/


theorem exists_locallyConstant_finite_nonempty {α : Type*} [Finite α] [Nonempty α]
    (hC : IsLimit C) (f : LocallyConstant C.pt α) :
    ∃ (j : J) (g : LocallyConstant (F.obj j) α), f = g.comap (C.π.app _) := by
  /-
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : Nonempty α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  inhabit α
  /-
    J : Type v
    inst✝³ : CategoryTheory.SmallCategory J
    inst✝² : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    inst✝¹ : Finite α
    inst✝ : Nonempty α
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    inhabited_h : Inhabited α
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  obtain ⟨j, gg, h⟩ := exists_locallyConstant_finite_aux _ hC f
  classical
  let ι : α → α → Fin 2 := fun a b => if a = b then 0 else 1
  let σ : (α → Fin 2) → α := fun f => if h : ∃ a : α, ι a = f then h.choose else default
  refine ⟨j, gg.map σ, ?_⟩
  ext x
  simp only [Functor.const_obj_obj, LocallyConstant.coe_comap, LocallyConstant.map_apply,
    Function.comp_apply]
  dsimp [σ]
  have h1 : ι (f x) = gg (C.π.app j x) := by
    change f.map (fun a b => if a = b then (0 : Fin 2) else 1) x = _
    rw [h]
    rfl
  have h2 : ∃ a : α, ι a = gg (C.π.app j x) := ⟨f x, h1⟩
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [dif_pos h2]
  apply_fun ι
  · rw [h2.choose_spec]
    exact h1
  · intro a b hh
    have hhh := congr_fun hh a
    dsimp [ι] at hhh
    rw [if_pos rfl] at hhh
    split_ifs at hhh with hh1
    · exact hh1.symm
    · exact False.elim (bot_ne_top hhh)


/-- Any locally constant function from a cofiltered limit of profinite sets factors through
one of the components. -/
theorem exists_locallyConstant {α : Type*} (hC : IsLimit C) (f : LocallyConstant C.pt α) :
    ∃ (j : J) (g : LocallyConstant (F.obj j) α), f = g.comap (C.π.app _) := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  let S := f.discreteQuotient
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  let ff : S → α := f.lift
  /-
    J : Type v
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofiltered J
    F : CategoryTheory.Functor J Profinite
    C : CategoryTheory.Limits.Cone F
    α : Type u_1
    hC : CategoryTheory.Limits.IsLimit C
    f : LocallyConstant (↑C.pt.toTop) α
    S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
    ff : Quotient S.toSetoid → α := ⇑f.lift
    ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
  -/
  cases isEmpty_or_nonempty S
  · suffices ∃ j, IsEmpty (F.obj j) by
      refine this.imp fun j hj => ?_
      refine ⟨⟨hj.elim, fun A => ?_⟩, ?_⟩
      · suffices (fun a ↦ IsEmpty.elim hj a) ⁻¹' A = ∅ by
          rw [this]
          exact isOpen_empty
        exact @Set.eq_empty_of_isEmpty _ hj _
      · ext x
        exact hj.elim' (C.π.app j x)
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      ⊢ Exists fun j => IsEmpty ↑(F.obj j).toTop
    -/
    simp only [← not_nonempty_iff, ← not_forall]
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      ⊢ Not (∀ (x : J), Nonempty ↑(F.obj x).toTop)
    -/
    intro h
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      h : ∀ (x : J), Nonempty ↑(F.obj x).toTop
      ⊢ False
    -/
    haveI : ∀ j : J, Nonempty ((F ⋙ Profinite.toTopCat).obj j) := h
    haveI : ∀ j : J, T2Space ((F ⋙ Profinite.toTopCat).obj j) := fun j =>
      (inferInstance : T2Space (F.obj j))
    haveI : ∀ j : J, CompactSpace ((F ⋙ Profinite.toTopCat).obj j) := fun j =>
      (inferInstance : CompactSpace (F.obj j))
    have cond := TopCat.nonempty_limitCone_of_compact_t2_cofiltered_system.{u}
      (F ⋙ Profinite.toTopCat)
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      h : ∀ (x : J), Nonempty ↑(F.obj x).toTop
      this✝¹ : ∀ (j : J), Nonempty ↑((F.comp Profinite.toTopCat).obj j)
      this✝ : ∀ (j : J), T2Space ↑((F.comp Profinite.toTopCat).obj j)
      this : ∀ (j : J), CompactSpace ↑((F.comp Profinite.toTopCat).obj j)
      cond : Nonempty ↑(TopCat.limitCone (F.comp Profinite.toTopCat)).pt
      ⊢ False
    -/
    suffices Nonempty C.pt from IsEmpty.false (S.proj this.some)
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      h : ∀ (x : J), Nonempty ↑(F.obj x).toTop
      this✝¹ : ∀ (j : J), Nonempty ↑((F.comp Profinite.toTopCat).obj j)
      this✝ : ∀ (j : J), T2Space ↑((F.comp Profinite.toTopCat).obj j)
      this : ∀ (j : J), CompactSpace ↑((F.comp Profinite.toTopCat).obj j)
      cond : Nonempty ↑(TopCat.limitCone (F.comp Profinite.toTopCat)).pt
      ⊢ Nonempty ↑C.pt.toTop
    -/
    let D := Profinite.toTopCat.mapCone C
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      h : ∀ (x : J), Nonempty ↑(F.obj x).toTop
      this✝¹ : ∀ (j : J), Nonempty ↑((F.comp Profinite.toTopCat).obj j)
      this✝ : ∀ (j : J), T2Space ↑((F.comp Profinite.toTopCat).obj j)
      this : ∀ (j : J), CompactSpace ↑((F.comp Profinite.toTopCat).obj j)
      cond : Nonempty ↑(TopCat.limitCone (F.comp Profinite.toTopCat)).pt
      D : CategoryTheory.Limits.Cone (F.comp Profinite.toTopCat) := Profinite.toTopC …
      ⊢ Nonempty ↑C.pt.toTop
    -/
    have hD : IsLimit D := isLimitOfPreserves Profinite.toTopCat hC
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      h : ∀ (x : J), Nonempty ↑(F.obj x).toTop
      this✝¹ : ∀ (j : J), Nonempty ↑((F.comp Profinite.toTopCat).obj j)
      this✝ : ∀ (j : J), T2Space ↑((F.comp Profinite.toTopCat).obj j)
      this : ∀ (j : J), CompactSpace ↑((F.comp Profinite.toTopCat).obj j)
      cond : Nonempty ↑(TopCat.limitCone (F.comp Profinite.toTopCat)).pt
      D : CategoryTheory.Limits.Cone (F.comp Profinite.toTopCat) := Profinite.toTopC …
      hD : CategoryTheory.Limits.IsLimit D
      ⊢ Nonempty ↑C.pt.toTop
    -/
    have CD := (hD.conePointUniqueUpToIso (TopCat.limitConeIsLimit.{v, max u v} _)).inv
    /-
      case inl
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : IsEmpty (Quotient S.toSetoid)
      h : ∀ (x : J), Nonempty ↑(F.obj x).toTop
      this✝¹ : ∀ (j : J), Nonempty ↑((F.comp Profinite.toTopCat).obj j)
      this✝ : ∀ (j : J), T2Space ↑((F.comp Profinite.toTopCat).obj j)
      this : ∀ (j : J), CompactSpace ↑((F.comp Profinite.toTopCat).obj j)
      cond : Nonempty ↑(TopCat.limitCone (F.comp Profinite.toTopCat)).pt
      D : CategoryTheory.Limits.Cone (F.comp Profinite.toTopCat) := Profinite.toTopC …
      hD : CategoryTheory.Limits.IsLimit D
      CD : Quiver.Hom (TopCat.limitCone (F.comp Profinite.toTopCat)).pt D.pt
      ⊢ Nonempty ↑C.pt.toTop
    -/
    exact cond.map CD
    /-
      🎉 no goals
    -/
    /-
      case inr
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
    -/
  · let f' : LocallyConstant C.pt S := ⟨S.proj, S.proj_isLocallyConstant⟩
    /-
      case inr
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
    -/
    obtain ⟨j, g', hj⟩ := exists_locallyConstant_finite_nonempty _ hC f'
    /-
      case inr.intro.intro
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      j : J
      g' : LocallyConstant (↑(F.obj j).toTop) (Quotient S.toSetoid)
      hj : Eq f' (LocallyConstant.comap (C.π.app j) g')
      ⊢ Exists fun j => Exists fun g => Eq f (LocallyConstant.comap (C.π.app j) g)
    -/
    refine ⟨j, ⟨ff ∘ g', g'.isLocallyConstant.comp _⟩, ?_⟩
    /-
      case inr.intro.intro
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      j : J
      g' : LocallyConstant (↑(F.obj j).toTop) (Quotient S.toSetoid)
      hj : Eq f' (LocallyConstant.comap (C.π.app j) g')
      ⊢ Eq f (LocallyConstant.comap (C.π.app j) { toFun := Function.comp ff ⇑g', isL …
    -/
    ext1 t
    /-
      case inr.intro.intro.h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      j : J
      g' : LocallyConstant (↑(F.obj j).toTop) (Quotient S.toSetoid)
      hj : Eq f' (LocallyConstant.comap (C.π.app j) g')
      t : ↑C.pt.toTop
      ⊢ Eq (f t) ((LocallyConstant.comap (C.π.app j) { toFun := Function.comp ff ⇑g' …
    -/
    apply_fun fun e => e t at hj
    /-
      case inr.intro.intro.h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      j : J
      g' : LocallyConstant (↑(F.obj j).toTop) (Quotient S.toSetoid)
      t : ↑C.pt.toTop
      hj : Eq (f' t) ((LocallyConstant.comap (C.π.app j) g') t)
      ⊢ Eq (f t) ((LocallyConstant.comap (C.π.app j) { toFun := Function.comp ff ⇑g' …
    -/
    dsimp at hj ⊢
    /-
      case inr.intro.intro.h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      j : J
      g' : LocallyConstant (↑(F.obj j).toTop) (Quotient S.toSetoid)
      t : ↑C.pt.toTop
      hj : Eq (f' t) (g' ((C.π.app j) t))
      ⊢ Eq (f t) (ff (g' ((C.π.app j) t)))
    -/
    rw [← hj]
    /-
      case inr.intro.intro.h
      J : Type v
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsCofiltered J
      F : CategoryTheory.Functor J Profinite
      C : CategoryTheory.Limits.Cone F
      α : Type u_1
      hC : CategoryTheory.Limits.IsLimit C
      f : LocallyConstant (↑C.pt.toTop) α
      S : DiscreteQuotient ↑C.pt.toTop := f.discreteQuotient
      ff : Quotient S.toSetoid → α := ⇑f.lift
      h✝ : Nonempty (Quotient S.toSetoid)
      f' : LocallyConstant (↑C.pt.toTop) (Quotient S.toSetoid) := { toFun := S.proj, …
      j : J
      g' : LocallyConstant (↑(F.obj j).toTop) (Quotient S.toSetoid)
      t : ↑C.pt.toTop
      hj : Eq (f' t) (g' ((C.π.app j) t))
      ⊢ Eq (f t) (ff (f' t))
    -/
    rfl
    /-
      🎉 no goals
    -/


