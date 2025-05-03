/-- Any `StrictSegal` simplicial set is a `Quasicategory`. -/
instance quasicategory {X : SSet.{u}} [StrictSegal X] : Quasicategory X := by
  /-
    X : SSet
    inst✝ : X.StrictSegal
    ⊢ X.Quasicategory
  -/
  apply quasicategory_of_filler X
  /-
    X : SSet
    inst✝ : X.StrictSegal
    ⊢ ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd …
  -/
  intro n i σ₀ h₀ hₙ
  /-
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    ⊢ Exists fun σ => ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory …
  -/
  use spineToSimplex <| Path.map (horn.spineId i h₀ hₙ) σ₀
  /-
    case h
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    ⊢ ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory.SimplicialObjec …
  -/
  intro j hj
  /-
    case h
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    j : Fin (HAdd.hAdd n 3)
    hj : Ne j i
    ⊢ Eq (CategoryTheory.SimplicialObject.δ X j (SSet.StrictSegal.spineToSimplex ( …
  -/
  apply spineInjective
  /-
    case h.a
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    j : Fin (HAdd.hAdd n 3)
    hj : Ne j i
    ⊢ Eq ((SSet.StrictSegal.spineEquiv (HAdd.hAdd n 1)) (CategoryTheory.Simplicial …
  -/
  ext k
  /-
    case h.a.h
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    j : Fin (HAdd.hAdd n 3)
    hj : Ne j i
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (((SSet.StrictSegal.spineEquiv (HAdd.hAdd n 1)) (CategoryTheory.Simplicia …
  -/
  dsimp only [spineEquiv, spine_arrow, Function.comp_apply, Equiv.coe_fn_mk]
  /-
    case h.a.h
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    j : Fin (HAdd.hAdd n 3)
    hj : Ne j i
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (CategoryTheory.SimplicialObject.δ …
  -/
  rw [← types_comp_apply (σ₀.app _) (X.map _), ← σ₀.naturality]
  /-
    case h.a.h
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    j : Fin (HAdd.hAdd n 3)
    hj : Ne j i
    k : Fin (HAdd.hAdd n 1)
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (CategoryTheory.SimplicialObject.δ …
  -/
  let ksucc := k.succ.castSucc
  /-
    case h.a.h
    X : SSet
    inst✝ : X.StrictSegal
    n : Nat
    i : Fin (HAdd.hAdd n 3)
    σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
    h₀ : LT.lt 0 i
    hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
    j : Fin (HAdd.hAdd n 3)
    hj : Ne j i
    k : Fin (HAdd.hAdd n 1)
    ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
    ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (CategoryTheory.SimplicialObject.δ …
  -/
  obtain hlt | hgt | heq : ksucc < j ∨ j < ksucc ∨ j = ksucc := by omega
    /-
      case h.a.h.inl
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hlt : LT.lt ksucc j
      ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (CategoryTheory.SimplicialObject.δ …
    -/
  · rw [← spine_arrow, spine_δ_arrow_lt _ hlt]
    /-
      case h.a.h.inl
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hlt : LT.lt ksucc j
      ⊢ Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).arrow k.castSucc) (CategoryTheory.C …
    -/
    dsimp only [Path.map, spine_arrow, Fin.coe_eq_castSucc]
    /-
      case h.a.h.inl
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hlt : LT.lt ksucc j
      ⊢ Eq (σ₀.app { unop := SimplexCategory.mk 1 } ((SSet.horn.spineId i h₀ hₙ).arr …
    -/
    apply congr_arg
    simp only [horn, horn.spineId, standardSimplex, uliftFunctor, Functor.comp_obj,
      yoneda_obj_obj, whiskering_obj_obj_map, uliftFunctor_map, yoneda_obj_map,
      standardSimplex.objEquiv, Equiv.ulift, Equiv.coe_fn_symm_mk,
      Quiver.Hom.unop_op, horn.face_coe, Subtype.mk.injEq]
    /-
      case h.a.h.inl.h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hlt : LT.lt ksucc j
      ⊢ Eq ((SSet.standardSimplex.spineId (HAdd.hAdd n 2)).arrow k.castSucc) { down  …
    -/
    rw [mkOfSucc_δ_lt hlt]
    /-
      case h.a.h.inl.h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hlt : LT.lt ksucc j
      ⊢ Eq ((SSet.standardSimplex.spineId (HAdd.hAdd n 2)).arrow k.castSucc) { down  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.a.h.inr.inl
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hgt : LT.lt j ksucc
      ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (CategoryTheory.SimplicialObject.δ …
    -/
  · rw [← spine_arrow, spine_δ_arrow_gt _ hgt]
    /-
      case h.a.h.inr.inl
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hgt : LT.lt j ksucc
      ⊢ Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).arrow k.succ) (CategoryTheory.Categ …
    -/
    dsimp only [Path.map, spine_arrow, Fin.coe_eq_castSucc]
    /-
      case h.a.h.inr.inl
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hgt : LT.lt j ksucc
      ⊢ Eq (σ₀.app { unop := SimplexCategory.mk 1 } ((SSet.horn.spineId i h₀ hₙ).arr …
    -/
    apply congr_arg
    simp only [horn, horn.spineId, standardSimplex, uliftFunctor, Functor.comp_obj,
      yoneda_obj_obj, whiskering_obj_obj_map, uliftFunctor_map, yoneda_obj_map,
      standardSimplex.objEquiv, Equiv.ulift, Equiv.coe_fn_symm_mk,
      Quiver.Hom.unop_op, horn.face_coe, Subtype.mk.injEq]
    /-
      case h.a.h.inr.inl.h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hgt : LT.lt j ksucc
      ⊢ Eq ((SSet.standardSimplex.spineId (HAdd.hAdd n 2)).arrow k.succ) { down := C …
    -/
    rw [mkOfSucc_δ_gt hgt]
    /-
      case h.a.h.inr.inl.h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      hgt : LT.lt j ksucc
      ⊢ Eq ((SSet.standardSimplex.spineId (HAdd.hAdd n 2)).arrow k.succ) { down := S …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · /- The only inner horn of `Δ[2]` does not contain the diagonal edge. -/
    have hn0 : n ≠ 0 := by
      rintro rfl
      obtain rfl : k = 0 := by omega
      fin_cases i <;> contradiction
    /- We construct the triangle in the standard simplex as a 2-simplex in
    the horn. While the triangle is not contained in the inner horn `Λ[2, 1]`,
    we can inhabit `Λ[n + 2, i] _[2]` by induction on `n`. -/
    let triangle : Λ[n + 2, i] _[2] := by
      cases n with
      | zero => contradiction
      | succ _ => exact horn.primitiveTriangle i h₀ hₙ k (by omega)
    /- The interval spanning from `k` to `k + 2` is equivalently the spine
    of the triangle with vertices `k`, `k + 1`, and `k + 2`. -/
    have hi : ((horn.spineId i h₀ hₙ).map σ₀).interval k 2 (by omega) =
        X.spine 2 (σ₀.app _ triangle) := by
      ext m
      dsimp [spine_arrow, Path.interval, Path.map]
      rw [← types_comp_apply (σ₀.app _) (X.map _), ← σ₀.naturality]
      apply congr_arg
      simp only [horn, standardSimplex, uliftFunctor, Functor.comp_obj,
        whiskering_obj_obj_obj, yoneda_obj_obj, uliftFunctor_obj, ne_eq,
        whiskering_obj_obj_map, uliftFunctor_map, yoneda_obj_map, len_mk,
        Nat.reduceAdd, Quiver.Hom.unop_op]
      cases n with
      | zero => contradiction
      | succ _ => ext x; fin_cases x <;> fin_cases m <;> rfl
    /-
      case h.a.h.inr.inr
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      heq : Eq j ksucc
      hn0 : Ne n 0
      triangle : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 2 } :=
        Nat.casesAuxOn (motive := fun a => Eq n a → (SSet.horn (HAdd.hAdd n 2) i).ob …
          (fun h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => absurd ⋯ hn0)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          (fun n_1 h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => SSet.horn.primitiveTriangle i h₀ hₙ ↑k ⋯)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          ⋯
      hi : Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).interval (↑k) 2 ⋯) (X.spine 2 (σ …
      ⊢ Eq (X.map (SimplexCategory.mkOfSucc k).op (CategoryTheory.SimplicialObject.δ …
    -/
    rw [← spine_arrow, spine_δ_arrow_eq _ heq, hi]
    /-
      case h.a.h.inr.inr
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      heq : Eq j ksucc
      hn0 : Ne n 0
      triangle : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 2 } :=
        Nat.casesAuxOn (motive := fun a => Eq n a → (SSet.horn (HAdd.hAdd n 2) i).ob …
          (fun h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => absurd ⋯ hn0)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          (fun n_1 h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => SSet.horn.primitiveTriangle i h₀ hₙ ↑k ⋯)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          ⋯
      hi : Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).interval (↑k) 2 ⋯) (X.spine 2 (σ …
      ⊢ Eq (SSet.StrictSegal.spineToDiagonal (X.spine 2 (σ₀.app { unop := SimplexCat …
    -/
    simp only [spineToDiagonal, diagonal, spineToSimplex_spine]
    /-
      case h.a.h.inr.inr
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      heq : Eq j ksucc
      hn0 : Ne n 0
      triangle : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 2 } :=
        Nat.casesAuxOn (motive := fun a => Eq n a → (SSet.horn (HAdd.hAdd n 2) i).ob …
          (fun h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => absurd ⋯ hn0)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          (fun n_1 h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => SSet.horn.primitiveTriangle i h₀ hₙ ↑k ⋯)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          ⋯
      hi : Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).interval (↑k) 2 ⋯) (X.spine 2 (σ …
      ⊢ Eq (X.map (SimplexCategory.diag 2).op (σ₀.app { unop := SimplexCategory.mk 2 …
    -/
    rw [← types_comp_apply (σ₀.app _) (X.map _), ← σ₀.naturality, types_comp_apply]
    /-
      case h.a.h.inr.inr
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      heq : Eq j ksucc
      hn0 : Ne n 0
      triangle : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 2 } :=
        Nat.casesAuxOn (motive := fun a => Eq n a → (SSet.horn (HAdd.hAdd n 2) i).ob …
          (fun h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => absurd ⋯ hn0)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          (fun n_1 h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => SSet.horn.primitiveTriangle i h₀ hₙ ↑k ⋯)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          ⋯
      hi : Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).interval (↑k) 2 ⋯) (X.spine 2 (σ …
      ⊢ Eq (σ₀.app { unop := SimplexCategory.mk 1 } ((SSet.horn (HAdd.hAdd n 2) i).m …
    -/
    apply congr_arg
    simp only [horn, standardSimplex, uliftFunctor, Functor.comp_obj,
      whiskering_obj_obj_obj, yoneda_obj_obj, uliftFunctor_obj,
      uliftFunctor_map, whiskering_obj_obj_map, yoneda_obj_map, horn.face_coe,
      len_mk, Nat.reduceAdd, Quiver.Hom.unop_op, Subtype.mk.injEq, ULift.up_inj]
    /-
      case h.a.h.inr.inr.h
      X : SSet
      inst✝ : X.StrictSegal
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      j : Fin (HAdd.hAdd n 3)
      hj : Ne j i
      k : Fin (HAdd.hAdd n 1)
      ksucc : Fin (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd n 1) 1) 1) := k.succ.castSucc
      heq : Eq j ksucc
      hn0 : Ne n 0
      triangle : (SSet.horn (HAdd.hAdd n 2) i).obj { unop := SimplexCategory.mk 2 } :=
        Nat.casesAuxOn (motive := fun a => Eq n a → (SSet.horn (HAdd.hAdd n 2) i).ob …
          (fun h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => absurd ⋯ hn0)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          (fun n_1 h =>
            Eq.ndrec (motive := fun ⦃n⦄ =>
              ⦃i : Fin (HAdd.hAdd n 3)⦄ →
                Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) X →
                  LT.lt 0 i →
                    LT.lt i (Fin.last (HAdd.hAdd n 2)) →
                      (j : Fin (HAdd.hAdd n 3)) →
                        Ne j i →
                          (k : Fin (HAdd.hAdd n 1)) →
                            let ksucc := k.succ.castSucc;
                            Eq j ksucc → Ne n 0 → (SSet.horn (HAdd.hAdd n 2) i).obj  …
              (fun ⦃i⦄ σ₀ h₀ hₙ j hj k =>
                let ksucc := k.succ.castSucc;
                fun heq hn0 => SSet.horn.primitiveTriangle i h₀ hₙ ↑k ⋯)
              ⋯ σ₀ h₀ hₙ j hj k heq hn0)
          ⋯
      hi : Eq (((SSet.horn.spineId i h₀ hₙ).map σ₀).interval (↑k) 2 ⋯) (X.spine 2 (σ …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplexCategory.diag 2) (↑triangle). …
    -/
    ext z
    cases n with
    | zero => contradiction
    | succ _ =>
      fin_cases z <;>
      · simp only [standardSimplex.objEquiv, uliftFunctor_map, yoneda_obj_map,
          Quiver.Hom.unop_op, Equiv.ulift_symm_down]
        rw [mkOfSucc_δ_eq heq]
        rfl


