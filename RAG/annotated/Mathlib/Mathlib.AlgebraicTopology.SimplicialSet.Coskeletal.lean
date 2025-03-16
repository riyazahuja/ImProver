/-- The identity natural transformation exhibits a simplicial set as a right extension of its
restriction along `(Truncated.inclusion (n := n)).op`.-/
@[simps!]
def rightExtensionInclusion (X : SSet.{u}) (n : ℕ) :
    RightExtension (Truncated.inclusion (n := n)).op
      ((Truncated.inclusion n).op ⋙ X) := RightExtension.mk _ (𝟙 _)


local notation (priority := high) "[" n "]" => SimplexCategory.mk n


local macro:max (priority := high) "[" n:term "]₂" : term =>
  `((⟨SimplexCategory.mk $n, by dsimp; omega⟩ : SimplexCategory.Truncated 2))


/-- A morphism in `SimplexCategory` with domain `[0]`, `[1]`, or `[2]` defines an object in the
comma category `StructuredArrow (op [n]) (Truncated.inclusion (n := 2)).op`.-/
abbrev strArrowMk₂ {i : ℕ} {n : ℕ} (φ : [i] ⟶ [n]) (hi : i ≤ 2) :
    StructuredArrow (op [n]) (Truncated.inclusion (n := 2)).op :=
                                             /-
                                               X : SSet
                                               inst✝ : X.StrictSegal
                                               i n : Nat
                                               φ : Quiver.Hom (SimplexCategory.mk i) (SimplexCategory.mk n)
                                               hi : LE.le i 2
                                               ⊢ Quiver.Hom { unop := SimplexCategory.mk n } ((SimplexCategory.Truncated.incl …
                                             -/
  StructuredArrow.mk (Y := op ⟨[i], hi⟩) (by exact φ.op)
                                             /-
                                               🎉 no goals
                                             -/


/-- Given a term in the cone over the diagram
`(proj (op [n]) ((Truncated.inclusion 2).op ⋙ (Truncated.inclusion 2).op ⋙ X)` where `X` is
Strict Segal, one can produce an `n`-simplex in `X`. -/
@[simp]
noncomputable def lift {X : SSet.{u}} [StrictSegal X] {n}
    (s : Cone (proj (op [n]) (Truncated.inclusion 2).op ⋙
      (Truncated.inclusion 2).op ⋙ X)) (x : s.pt) : X _[n] :=
  StrictSegal.spineToSimplex {
    vertex := fun i ↦ s.π.app (.mk (Y := op [0]₂) (.op (SimplexCategory.const _ _ i))) x
    arrow := fun i ↦ s.π.app (.mk (Y := op [1]₂) (.op (mkOfLe _ _ (Fin.castSucc_le_succ i)))) x
    arrow_src := fun i ↦ by
      let φ : strArrowMk₂ (mkOfLe _ _ (Fin.castSucc_le_succ i)) (by simp) ⟶
        strArrowMk₂ ([0].const _ i.castSucc) (by simp) :=
          StructuredArrow.homMk (δ 1).op
          (Quiver.Hom.unop_inj (by ext x; fin_cases x; rfl))
      /-
        X✝ : SSet
        inst✝¹ : X✝.StrictSegal
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Fin n
        φ : Quiver.Hom (SSet.StrictSegal.isPointwiseRightKanExtensionAt.strArrowMk₂ (S …
        ⊢ Eq (CategoryTheory.SimplicialObject.δ X 1 ((fun i => s.π.app (CategoryTheory …
      -/
      exact congr_fun (s.w φ) x
      /-
        🎉 no goals
      -/
    arrow_tgt := fun i ↦ by
      /-
        X✝ : SSet
        inst✝¹ : X✝.StrictSegal
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Fin n
        ⊢ Eq (CategoryTheory.SimplicialObject.δ X 0 ((fun i => s.π.app (CategoryTheory …
      -/
      dsimp
      let φ : strArrowMk₂ (mkOfLe _ _ (Fin.castSucc_le_succ i)) (by simp) ⟶
          strArrowMk₂ ([0].const _ i.succ) (by simp) :=
        StructuredArrow.homMk (δ 0).op
          (Quiver.Hom.unop_inj (by ext x; fin_cases x; rfl))
      /-
        X✝ : SSet
        inst✝¹ : X✝.StrictSegal
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Fin n
        φ : Quiver.Hom (SSet.StrictSegal.isPointwiseRightKanExtensionAt.strArrowMk₂ (S …
        ⊢ Eq (CategoryTheory.SimplicialObject.δ X 0 (s.π.app (CategoryTheory.Structure …
      -/
      exact congr_fun (s.w φ) x }
      /-
        🎉 no goals
      -/


lemma fac_aux₁ {n : ℕ}
    (s : Cone (proj (op [n]) (Truncated.inclusion 2).op ⋙ (Truncated.inclusion 2).op ⋙ X))
    (x : s.pt) (i : ℕ) (hi : i < n) :
    X.map (mkOfSucc ⟨i, hi⟩).op (lift s x) =
                                                  /-
                                                    X : SSet
                                                    inst✝ : X.StrictSegal
                                                    n : Nat
                                                    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
                                                    x : s.pt
                                                    i : Nat
                                                    hi : LT.lt i n
                                                    ⊢ LE.le 1 2
                                                  -/
      s.π.app (strArrowMk₂ (mkOfSucc ⟨i, hi⟩) (by omega)) x := by
                                                  /-
                                                    🎉 no goals
                                                  -/
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
    x : s.pt
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc ⟨i, hi⟩).op (SSet.StrictSegal.isPointwis …
  -/
  dsimp [lift]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
    x : s.pt
    i : Nat
    hi : LT.lt i n
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc ⟨i, hi⟩).op (SSet.StrictSegal.spineToSim …
  -/
  rw [spineToSimplex_arrow]
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
    x : s.pt
    i : Nat
    hi : LT.lt i n
    ⊢ Eq ({ vertex := fun i => s.π.app (CategoryTheory.StructuredArrow.mk (((Simpl …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma fac_aux₂ {n : ℕ}
    (s : Cone (proj (op [n]) (Truncated.inclusion 2).op ⋙ (Truncated.inclusion 2).op ⋙ X))
    (x : s.pt) (i j : ℕ) (hij : i ≤ j) (hj : j ≤ n) :
                         /-
                           X : SSet
                           inst✝ : X.StrictSegal
                           n : Nat
                           s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
                           x : s.pt
                           i j : Nat
                           hij : LE.le i j
                           hj : LE.le j n
                           ⊢ LT.lt i (HAdd.hAdd n 1)
                         -/
                         /-
                           🎉 no goals
                         -/
    X.map (mkOfLe ⟨i, by omega⟩ ⟨j, by omega⟩ hij).op (lift s x) =
                                       /-
                                         🎉 no goals
                                       -/
                                          /-
                                            X : SSet
                                            inst✝ : X.StrictSegal
                                            n : Nat
                                            s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
                                            x : s.pt
                                            i j : Nat
                                            hij : LE.le i j
                                            hj : LE.le j n
                                            ⊢ LT.lt i (HAdd.hAdd n 1)
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                                        /-
                                                          🎉 no goals
                                                        -/
      s.π.app (strArrowMk₂ (mkOfLe ⟨i, by omega⟩ ⟨j, by omega⟩ hij) (by omega)) x := by
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
    x : s.pt
    i j : Nat
    hij : LE.le i j
    hj : LE.le j n
    ⊢ Eq (X.map (SimplexCategory.mkOfLe ⟨i, ⋯⟩ ⟨j, ⋯⟩ hij).op (SSet.StrictSegal.is …
  -/
  obtain ⟨k, hk⟩ := Nat.le.dest hij
  /-
    case intro
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
    x : s.pt
    i j : Nat
    hij : LE.le i j
    hj : LE.le j n
    k : Nat
    hk : Eq (HAdd.hAdd i k) j
    ⊢ Eq (X.map (SimplexCategory.mkOfLe ⟨i, ⋯⟩ ⟨j, ⋯⟩ hij).op (SSet.StrictSegal.is …
  -/
  revert i j
  induction k with
  | zero =>
      rintro i j hij hj hik
      obtain rfl : i = j := by omega
      have : mkOfLe ⟨i, Nat.lt_add_one_of_le hj⟩ ⟨i, Nat.lt_add_one_of_le hj⟩ (by omega) =
        [1].const [0] 0 ≫ [0].const [n] ⟨i, Nat.lt_add_one_of_le hj⟩ := Hom.ext_one_left _ _
      rw [this]
      let α : (strArrowMk₂ ([0].const [n] ⟨i, Nat.lt_add_one_of_le hj⟩) (by omega)) ⟶
        (strArrowMk₂ ([1].const [0] 0 ≫ [0].const [n] ⟨i, Nat.lt_add_one_of_le hj⟩) (by omega)) :=
            StructuredArrow.homMk (([1].const [0] 0).op) (by simp; rfl)
      have nat := congr_fun (s.π.naturality α) x
      dsimp only [Fin.val_zero, Nat.add_zero, id_eq, Int.reduceNeg, Int.Nat.cast_ofNat_Int,
        Int.reduceAdd, Fin.eta, comp_obj, StructuredArrow.proj_obj, op_obj, const_obj_obj,
        const_obj_map, types_comp_apply, types_id_apply, Functor.comp_map, StructuredArrow.proj_map,
        op_map] at nat
      rw [nat, op_comp, Functor.map_comp]
      simp only [types_comp_apply]
      refine congrArg (X.map ([1].const [0] 0).op) ?_
      unfold strArrowMk₂
      rw [lift, StrictSegal.spineToSimplex_vertex]
      congr
  | succ k hk =>
      intro i j hij hj hik
      let α := strArrowMk₂ (mkOfLeComp (n := n) ⟨i, by omega⟩ ⟨i + k, by omega⟩
          ⟨j, by omega⟩ (by simp)
        (by simp only [Fin.mk_le_mk]; omega)) (by rfl)
      let α₀ := strArrowMk₂ (mkOfLe (n := n) ⟨i + k, by omega⟩ ⟨j, by omega⟩
        (by simp only [Fin.mk_le_mk]; omega)) (by simp)
      let α₁ := strArrowMk₂ (mkOfLe (n := n) ⟨i, by omega⟩ ⟨j, by omega⟩
        (by simp only [Fin.mk_le_mk]; omega)) (by simp)
      let α₂ := strArrowMk₂ (mkOfLe (n := n) ⟨i, by omega⟩ ⟨i + k, by omega⟩ (by simp)) (by simp)
      let β₀ : α ⟶ α₀ := StructuredArrow.homMk ((mkOfSucc 1).op) (Quiver.Hom.unop_inj
        (by ext x; fin_cases x <;> rfl))
      let β₁ : α ⟶ α₁ := StructuredArrow.homMk ((δ 1).op) (Quiver.Hom.unop_inj
        (by ext x; fin_cases x <;> rfl))
      let β₂ : α ⟶ α₂ := StructuredArrow.homMk ((mkOfSucc 0).op) (Quiver.Hom.unop_inj
        (by ext x; fin_cases x <;> rfl))
      have h₀ : X.map α₀.hom (lift s x) = s.π.app α₀ x := by
        obtain rfl : j = (i + k) + 1 := by omega
        exact fac_aux₁ _ _ _ _ (by omega)
      have h₂ : X.map α₂.hom (lift s x) = s.π.app α₂ x :=
        hk i (i + k) (by simp) (by omega) rfl
      change X.map α₁.hom (lift s x) = s.π.app α₁ x
      have : X.map α.hom (lift s x) = s.π.app α x := by
        apply StrictSegal.spineInjective
        apply Path.ext'
        intro t
        dsimp only [spineEquiv]
        rw [Equiv.coe_fn_mk, spine_arrow, spine_arrow,
            ← FunctorToTypes.map_comp_apply]
        match t with
        | 0 =>
            have : α.hom ≫ (mkOfSucc 0).op = α₂.hom :=
              Quiver.Hom.unop_inj (by ext x ; fin_cases x <;> rfl)
            rw [this, h₂, ← congr_fun (s.w β₂) x]
            rfl
        | 1 =>
            have : α.hom ≫ (mkOfSucc 1).op = α₀.hom :=
              Quiver.Hom.unop_inj (by ext x ; fin_cases x <;> rfl)
            rw [this, h₀, ← congr_fun (s.w β₀) x]
            rfl
      rw [← StructuredArrow.w β₁, FunctorToTypes.map_comp_apply, this, ← s.w β₁]
      dsimp


lemma fac_aux₃ {n : ℕ}
    (s : Cone (proj (op [n]) (Truncated.inclusion 2).op ⋙ (Truncated.inclusion 2).op ⋙ X))
    (x : s.pt) (φ : [1] ⟶ [n]) :
                                                       /-
                                                         X : SSet
                                                         inst✝ : X.StrictSegal
                                                         n : Nat
                                                         s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
                                                         x : s.pt
                                                         φ : Quiver.Hom (SimplexCategory.mk 1) (SimplexCategory.mk n)
                                                         ⊢ LE.le 1 2
                                                       -/
    X.map φ.op (lift s x) = s.π.app (strArrowMk₂ φ (by omega)) x := by
                                                       /-
                                                         🎉 no goals
                                                       -/
  obtain ⟨i, j, hij, rfl⟩ : ∃ i j hij, φ = mkOfLe i j hij :=
    ⟨φ.toOrderHom 0, φ.toOrderHom 1, φ.toOrderHom.monotone (by simp),
      Hom.ext_one_left _ _ rfl rfl⟩
  /-
    case intro.intro.intro
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
    x : s.pt
    i j : Fin (HAdd.hAdd n 1)
    hij : LE.le i j
    ⊢ Eq (X.map (SimplexCategory.mkOfLe i j hij).op (SSet.StrictSegal.isPointwiseR …
  -/
  exact fac_aux₂ _ _ _ _ _ _ (by omega)
  /-
    🎉 no goals
  -/


open isPointwiseRightKanExtensionAt in
/-- A strict Segal simplicial set is 2-coskeletal. -/
noncomputable def isPointwiseRightKanExtensionAt (n : ℕ) :
    (rightExtensionInclusion X 2).IsPointwiseRightKanExtensionAt ⟨[n]⟩ where
  lift s x := lift (X := X) s x
  fac s j := by
    /-
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (SimplexCa …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s x => SSet.StrictSegal.isPoint …
    -/
    ext x
    /-
      case h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (SimplexCa …
      x : s.pt
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s x => SSet.StrictSegal.isPoint …
    -/
    obtain ⟨⟨i, hi⟩, ⟨f :  _ ⟶ _⟩, rfl⟩ := j.mk_surjective
    /-
      case h.intro.op.mk.intro.op
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      x : s.pt
      i : SimplexCategory
      hi : LE.le i.len 2
      f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s x => SSet.StrictSegal.isPoint …
    -/
    obtain ⟨i, rfl⟩ : ∃ j, SimplexCategory.mk j = i := ⟨_, i.mk_len⟩
    /-
      case h.intro.op.mk.intro.op.intro
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      x : s.pt
      i : Nat
      hi : LE.le (SimplexCategory.mk i).len 2
      f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s x => SSet.StrictSegal.isPoint …
    -/
    dsimp at hi ⊢
    /-
      case h.intro.op.mk.intro.op.intro
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      x : s.pt
      i : Nat
      hi : LE.le i 2
      f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
      ⊢ Eq (X.map { unop := f } (SSet.StrictSegal.spineToSimplex { vertex := fun i = …
    -/
    apply StrictSegal.spineInjective
    /-
      case h.intro.op.mk.intro.op.intro.a
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      x : s.pt
      i : Nat
      hi : LE.le i 2
      f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
      ⊢ Eq ((SSet.StrictSegal.spineEquiv i) (X.map { unop := f } (SSet.StrictSegal.s …
    -/
    dsimp
    /-
      case h.intro.op.mk.intro.op.intro.a
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      x : s.pt
      i : Nat
      hi : LE.le i 2
      f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
      ⊢ Eq ((SSet.StrictSegal.spineEquiv i) (X.map { unop := f } (SSet.StrictSegal.s …
    -/
    ext k
      /-
        case h.intro.op.mk.intro.op.intro.a.vertex.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin (HAdd.hAdd i 1)
        ⊢ Eq (((SSet.StrictSegal.spineEquiv i) (X.map { unop := f } (SSet.StrictSegal. …
      -/
    · dsimp only [spineEquiv, Equiv.coe_fn_mk]
      /-
        case h.intro.op.mk.intro.op.intro.a.vertex.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin (HAdd.hAdd i 1)
        ⊢ Eq ((X.spine i (X.map { unop := f } (SSet.StrictSegal.spineToSimplex { verte …
      -/
      erw [spine_map_vertex]
      /-
        case h.intro.op.mk.intro.op.intro.a.vertex.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin (HAdd.hAdd i 1)
        ⊢ Eq ((X.spine n (SSet.StrictSegal.spineToSimplex { vertex := fun i => s.π.app …
      -/
      rw [spine_spineToSimplex, spine_vertex]
      let α : strArrowMk₂ f hi ⟶ strArrowMk₂ ([0].const [n] (f.toOrderHom k)) (by omega) :=
        StructuredArrow.homMk (([0].const _ (by exact k)).op) (by simp; rfl)
      /-
        case h.intro.op.mk.intro.op.intro.a.vertex.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin (HAdd.hAdd i 1)
        α : Quiver.Hom (SSet.StrictSegal.isPointwiseRightKanExtensionAt.strArrowMk₂ f  …
        ⊢ Eq ({ vertex := fun i => s.π.app (CategoryTheory.StructuredArrow.mk (((Simpl …
      -/
      exact congr_fun (s.w α).symm x
      /-
        🎉 no goals
      -/
      /-
        case h.intro.op.mk.intro.op.intro.a.arrow.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin i
        ⊢ Eq (((SSet.StrictSegal.spineEquiv i) (X.map { unop := f } (SSet.StrictSegal. …
      -/
    · dsimp only [spineEquiv, Equiv.coe_fn_mk, spine_arrow]
      /-
        case h.intro.op.mk.intro.op.intro.a.arrow.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin i
        ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (X.map { unop := f } (SSet.StrictS …
      -/
      rw [← FunctorToTypes.map_comp_apply]
      let α : strArrowMk₂ f hi ⟶ strArrowMk₂ (mkOfSucc k ≫ f) (by omega) :=
        StructuredArrow.homMk (mkOfSucc k).op (by simp; rfl)
      /-
        case h.intro.op.mk.intro.op.intro.a.arrow.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        x : s.pt
        i : Nat
        hi : LE.le i 2
        f : Quiver.Hom (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj  …
        k : Fin i
        α : Quiver.Hom (SSet.StrictSegal.isPointwiseRightKanExtensionAt.strArrowMk₂ f  …
        ⊢ Eq (X.map (CategoryTheory.CategoryStruct.comp { unop := f } (SimplexCategory …
      -/
      exact (isPointwiseRightKanExtensionAt.fac_aux₃ _ _ _ _).trans (congr_fun (s.w α).symm x)
      /-
        🎉 no goals
      -/
  uniq s m hm := by
    /-
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
      hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
      ⊢ Eq m ((fun s x => SSet.StrictSegal.isPointwiseRightKanExtensionAt.lift s x) s)
    -/
    ext x
    /-
      case h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
      hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
      x : s.pt
      ⊢ Eq (m x) ((fun s x => SSet.StrictSegal.isPointwiseRightKanExtensionAt.lift s …
    -/
    apply StrictSegal.spineInjective (X := X)
    /-
      case h.a
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
      hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
      x : s.pt
      ⊢ Eq ((SSet.StrictSegal.spineEquiv n) (m x)) ((SSet.StrictSegal.spineEquiv n)  …
    -/
    dsimp [spineEquiv]
    /-
      case h.a
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
      hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
      x : s.pt
      ⊢ Eq (X.spine n (m x)) (X.spine n (SSet.StrictSegal.spineToSimplex { vertex := …
    -/
    rw [StrictSegal.spine_spineToSimplex]
    /-
      case h.a
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
      m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
      hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
      x : s.pt
      ⊢ Eq (X.spine n (m x)) { vertex := fun i => s.π.app (CategoryTheory.Structured …
    -/
    ext i
      /-
        case h.a.vertex.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
        hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
        x : s.pt
        i : Fin (HAdd.hAdd n 1)
        ⊢ Eq ((X.spine n (m x)).vertex i) ({ vertex := fun i => s.π.app (CategoryTheor …
      -/
    · exact congr_fun (hm (StructuredArrow.mk (Y := op [0]₂) ([0].const [n] i).op)) x
      /-
        🎉 no goals
      -/
      /-
        case h.a.arrow.h
        X : SSet
        inst✝ : X.StrictSegal
        n : Nat
        s : CategoryTheory.Limits.Cone ((CategoryTheory.StructuredArrow.proj { unop := …
        m : Quiver.Hom s.pt ((SSet.Truncated.rightExtensionInclusion X 2).coneAt { uno …
        hm : ∀ (j : CategoryTheory.StructuredArrow { unop := SimplexCategory.mk n } (S …
        x : s.pt
        i : Fin n
        ⊢ Eq ((X.spine n (m x)).arrow i) ({ vertex := fun i => s.π.app (CategoryTheory …
      -/
    · exact congr_fun (hm (.mk (Y := op [1]₂) (.op (mkOfLe _ _ (Fin.castSucc_le_succ i))))) x
      /-
        🎉 no goals
      -/


/-- Since `StrictSegal.isPointwiseRightKanExtensionAt` proves that the appropriate
cones are limit cones, `rightExtensionInclusion X 2` is a pointwise right Kan extension.-/
noncomputable def isPointwiseRightKanExtension :
    (rightExtensionInclusion X 2).IsPointwiseRightKanExtension :=
  fun Δ => isPointwiseRightKanExtensionAt X Δ.unop.len


theorem isRightKanExtension :
    X.IsRightKanExtension (𝟙 ((inclusion 2).op ⋙ X)) :=
  RightExtension.IsPointwiseRightKanExtension.isRightKanExtension
    (isPointwiseRightKanExtension X)


/-- When `X` is `StrictSegal`, `X` is 2-coskeletal. -/
instance isCoskeletal : SimplicialObject.IsCoskeletal X 2 where
  isRightKanExtension := isRightKanExtension X


/-- The essential data of the nerve functor is contained in the 2-truncation, which is
recorded by the composite functor `nerveFunctor₂`.-/
def nerveFunctor₂ : Cat.{v, u} ⥤ SSet.Truncated 2 := nerveFunctor ⋙ truncation 2


/-- The natural isomorphism between `nerveFunctor` and `nerveFunctor₂ ⋙ Truncated.cosk 2` whose
components `nerve C ≅ (Truncated.cosk 2).obj (nerveFunctor₂.obj C)` shows that nerves of categories
are 2-coskeletal.-/
noncomputable def cosk₂Iso : nerveFunctor.{v, u} ≅ nerveFunctor₂.{v, u} ⋙ Truncated.cosk 2 :=
  NatIso.ofComponents (fun C ↦ (nerve C).isoCoskOfIsCoskeletal 2)
    (fun _ ↦ (coskAdj 2).unit.naturality _)


