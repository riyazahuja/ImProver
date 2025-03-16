/-- An object in `Mat_ C` is a finite tuple of objects in `C`.
-/
structure Mat_ where
  ι : Type
  [fintype : Fintype ι]
  X : ι → C


/-- A morphism in `Mat_ C` is a dependently typed matrix of morphisms. -/
def Hom (M N : Mat_ C) : Type v₁ :=
  DMatrix M.ι N.ι fun i j => M.X i ⟶ N.X j


/-- The identity matrix consists of identity morphisms on the diagonal, and zeros elsewhere. -/
def id (M : Mat_ C) : Hom M M := fun i j => if h : i = j then eqToHom (congr_arg M.X h) else 0


/-- Composition of matrices using matrix multiplication. -/
def comp {M N K : Mat_ C} (f : Hom M N) (g : Hom N K) : Hom M K := fun i k =>
  ∑ j : N.ι, f i j ≫ g j k


attribute [local simp] Hom.id Hom.comp


instance : Category.{v₁} (Mat_ C) where
  Hom := Hom
  id := Hom.id
  comp f g := f.comp g
                  /-
                    C : Type u₁
                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝ : CategoryTheory.Preadditive C
                    X✝ Y✝ : CategoryTheory.Mat_ C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                  -/
  id_comp f := by simp (config := { unfoldPartialApp := true }) [dite_comp]
                  /-
                    🎉 no goals
                  -/
                  /-
                    C : Type u₁
                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                    inst✝ : CategoryTheory.Preadditive C
                    X✝ Y✝ : CategoryTheory.Mat_ C
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                  -/
  comp_id f := by simp (config := { unfoldPartialApp := true }) [comp_dite]
                  /-
                    🎉 no goals
                  -/
  assoc f g h := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Mat_ C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    apply DMatrix.ext
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Mat_ C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      ⊢ ∀ (i : W✝.ι) (j : Z✝.ι), Eq (CategoryTheory.CategoryStruct.comp (CategoryThe …
    -/
    intros
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Mat_ C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      i✝ : W✝.ι
      j✝ : Z✝.ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
    -/
    simp_rw [Hom.comp, sum_comp, comp_sum, Category.assoc]
    /-
      case a
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      W✝ X✝ Y✝ Z✝ : CategoryTheory.Mat_ C
      f : Quiver.Hom W✝ X✝
      g : Quiver.Hom X✝ Y✝
      h : Quiver.Hom Y✝ Z✝
      i✝ : W✝.ι
      j✝ : Z✝.ι
      ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun x_1 => CategoryTheory.Categ …
    -/
    rw [Finset.sum_comm]
    /-
      🎉 no goals
    -/


@[ext]
theorem hom_ext {M N : Mat_ C} (f g : M ⟶ N) (H : ∀ i j, f i j = g i j) : f = g :=
  DMatrix.ext_iff.mp H


theorem id_def (M : Mat_ C) :
    (𝟙 M : Hom M M) = fun i j => if h : i = j then eqToHom (congr_arg M.X h) else 0 :=
  rfl


theorem id_apply (M : Mat_ C) (i j : M.ι) :
    (𝟙 M : Hom M M) i j = if h : i = j then eqToHom (congr_arg M.X h) else 0 :=
  rfl


@[simp]
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 inst✝ : CategoryTheory.Preadditive C
                                                                                 M : CategoryTheory.Mat_ C
                                                                                 i : M.ι
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.id M i i) (CategoryTheory.CategoryStruct.i …
                                                                               -/
theorem id_apply_self (M : Mat_ C) (i : M.ι) : (𝟙 M : Hom M M) i i = 𝟙 _ := by simp [id_apply]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem id_apply_of_ne (M : Mat_ C) (i j : M.ι) (h : i ≠ j) : (𝟙 M : Hom M M) i j = 0 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    M : CategoryTheory.Mat_ C
    i j : M.ι
    h : Ne i j
    ⊢ Eq (CategoryTheory.CategoryStruct.id M i j) 0
  -/
  simp [id_apply, h]
  /-
    🎉 no goals
  -/


theorem comp_def {M N K : Mat_ C} (f : M ⟶ N) (g : N ⟶ K) :
    f ≫ g = fun i k => ∑ j : N.ι, f i j ≫ g j k :=
  rfl


@[simp]
theorem comp_apply {M N K : Mat_ C} (f : M ⟶ N) (g : N ⟶ K) (i k) :
    (f ≫ g) i k = ∑ j : N.ι, f i j ≫ g j k :=
  rfl


instance (M N : Mat_ C) : Inhabited (M ⟶ N) :=
  ⟨fun i j => (0 : M.X i ⟶ N.X j)⟩


instance (M N : Mat_ C) : AddCommGroup (M ⟶ N) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    M N : CategoryTheory.Mat_ C
    ⊢ AddCommGroup (Quiver.Hom M N)
  -/
  change AddCommGroup (DMatrix M.ι N.ι _)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    M N : CategoryTheory.Mat_ C
    ⊢ AddCommGroup (DMatrix M.ι N.ι fun i j => Quiver.Hom (M.X i) (N.X j))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem add_apply {M N : Mat_ C} (f g : M ⟶ N) (i j) : (f + g) i j = f i j + g i j :=
  rfl


instance : Preadditive (Mat_ C) where
                              /-
                                C : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                inst✝ : CategoryTheory.Preadditive C
                                M N K : CategoryTheory.Mat_ C
                                f f' : Quiver.Hom M N
                                g : Quiver.Hom N K
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd f f') g) (HAdd.hAdd (Categ …
                              -/
  add_comp M N K f f' g := by ext; simp [Finset.sum_add_distrib]
                                   /-
                                     🎉 no goals
                                   -/
                              /-
                                C : Type u₁
                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                inst✝ : CategoryTheory.Preadditive C
                                M N K : CategoryTheory.Mat_ C
                                f : Quiver.Hom M N
                                g g' : Quiver.Hom N K
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (HAdd.hAdd g g')) (HAdd.hAdd (Categ …
                              -/
  comp_add M N K f g g' := by ext; simp [Finset.sum_add_distrib]
                                   /-
                                     🎉 no goals
                                   -/


/-- We now prove that `Mat_ C` has finite biproducts.

Be warned, however, that `Mat_ C` is not necessarily Krull-Schmidt,
and so the internal indexing of a biproduct may have nothing to do with the external indexing,
even though the construction we give uses a sigma type.
See however `isoBiproductEmbedding`.
-/
instance hasFiniteBiproducts : HasFiniteBiproducts (Mat_ C) where
  out n :=
    { has_biproduct := fun f =>
        hasBiproduct_of_total
          { pt := ⟨Σ j, (f j).ι, fun p => (f p.1).X p.2⟩
            π := fun j x y => by
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                y : (f j).ι
                ⊢ (fun i j_1 => Quiver.Hom ((CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) f …
              -/
              refine if h : x.1 = j then ?_ else 0
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                y : (f j).ι
                h : Eq x.fst j
                ⊢ (fun i j_1 => Quiver.Hom ((CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) f …
              -/
              refine if h' : @Eq.ndrec (Fin n) x.1 (fun j => (f j).ι) x.2 _ h = y then ?_ else 0
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                y : (f j).ι
                h : Eq x.fst j
                h' : Eq (Eq.ndrec x.snd h) y
                ⊢ (fun i j_1 => Quiver.Hom ((CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) f …
              -/
              apply eqToHom
              /-
                case p
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                y : (f j).ι
                h : Eq x.fst j
                h' : Eq (Eq.ndrec x.snd h) y
                ⊢ Eq ((CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p. …
              -/
              substs h h'
              /-
                case p
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                x : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                ⊢ Eq ((CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p. …
              -/
              rfl
              /-
                🎉 no goals
              -/
            -- Notice we were careful not to use `subst` until we had a goal in `Prop`.
            ι := fun j x y => by
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (f j).ι
                y : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                ⊢ (fun i j_1 => Quiver.Hom ((f j).X i) ((CategoryTheory.Mat_.mk (Sigma fun j = …
              -/
              refine if h : y.1 = j then ?_ else 0
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (f j).ι
                y : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                h : Eq y.fst j
                ⊢ (fun i j_1 => Quiver.Hom ((f j).X i) ((CategoryTheory.Mat_.mk (Sigma fun j = …
              -/
              refine if h' : @Eq.ndrec _ y.1 (fun j => (f j).ι) y.2 _ h = x then ?_ else 0
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (f j).ι
                y : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                h : Eq y.fst j
                h' : Eq (Eq.ndrec y.snd h) x
                ⊢ (fun i j_1 => Quiver.Hom ((f j).X i) ((CategoryTheory.Mat_.mk (Sigma fun j = …
              -/
              apply eqToHom
              /-
                case p
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j : Fin n
                x : (f j).ι
                y : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                h : Eq y.fst j
                h' : Eq (Eq.ndrec y.snd h) x
                ⊢ Eq ((f j).X x) ((CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f …
              -/
              substs h h'
              /-
                case p
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                y : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst).X p.sn …
                ⊢ Eq ((f y.fst).X (Eq.ndrec y.snd ⋯)) ((CategoryTheory.Mat_.mk (Sigma fun j => …
              -/
              rfl
              /-
                🎉 no goals
              -/
            ι_π := fun j j' => by
              /-
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j j' : Fin n
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j x y => dite (Eq y.fst j) (fun …
              -/
              ext x y
              /-
                case H
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j j' : Fin n
                x : (f j).ι
                y : (f j').ι
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j x y => dite (Eq y.fst j) (fun …
              -/
              dsimp
              /-
                case H
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j j' : Fin n
                x : (f j).ι
                y : (f j').ι
                ⊢ Eq (Finset.univ.sum fun j_1 => CategoryTheory.CategoryStruct.comp (dite (Eq  …
              -/
              simp_rw [dite_comp, comp_dite]
              simp only [ite_self, dite_eq_ite, Limits.comp_zero, Limits.zero_comp,
                eqToHom_trans, Finset.sum_congr]
              /-
                case H
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j j' : Fin n
                x : (f j).ι
                y : (f j').ι
                ⊢ Eq (Finset.univ.sum fun x_1 => dite (Eq x_1.fst j) (fun h => dite (Eq (Eq.re …
              -/
              erw [Finset.sum_sigma]
              /-
                case H
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j j' : Fin n
                x : (f j).ι
                y : (f j').ι
                ⊢ Eq (Finset.univ.sum fun a => Finset.univ.sum fun s => dite (Eq ⟨a, s⟩.fst j) …
              -/
              dsimp
              simp only [if_true, Finset.sum_dite_irrel, Finset.mem_univ,
                Finset.sum_const_zero, Finset.sum_congr, Finset.sum_dite_eq']
              /-
                case H
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                j j' : Fin n
                x : (f j).ι
                y : (f j').ι
                ⊢ Eq (dite (Eq j j') (fun h => dite (Eq (Eq.rec x ⋯) y) (fun h_1 => CategoryTh …
              -/
              split_ifs with h h'
                /-
                  case pos
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  j j' : Fin n
                  x : (f j).ι
                  y : (f j').ι
                  h : Eq j j'
                  h' : Eq (Eq.rec x ⋯) y
                  ⊢ Eq (CategoryTheory.eqToHom ⋯) (CategoryTheory.eqToHom ⋯ x y)
                -/
              · substs h h'
                /-
                  case pos
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  j : Fin n
                  x : (f j).ι
                  ⊢ Eq (CategoryTheory.eqToHom ⋯) (CategoryTheory.eqToHom ⋯ x (Eq.rec x ⋯))
                -/
                simp only [CategoryTheory.eqToHom_refl, CategoryTheory.Mat_.id_apply_self]
                /-
                  🎉 no goals
                -/
                /-
                  case neg
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  j j' : Fin n
                  x : (f j).ι
                  y : (f j').ι
                  h : Eq j j'
                  h' : Not (Eq (Eq.rec x ⋯) y)
                  ⊢ Eq 0 (CategoryTheory.eqToHom ⋯ x y)
                -/
              · subst h
                /-
                  case neg
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  j : Fin n
                  x y : (f j).ι
                  h' : Not (Eq (Eq.rec x ⋯) y)
                  ⊢ Eq 0 (CategoryTheory.eqToHom ⋯ x y)
                -/
                rw [eqToHom_refl, id_apply_of_ne _ _ _ h']
                /-
                  🎉 no goals
                -/
                /-
                  case neg
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  j j' : Fin n
                  x : (f j).ι
                  y : (f j').ι
                  h : Not (Eq j j')
                  ⊢ Eq 0 (0 x y)
                -/
              · rfl }
                /-
                  🎉 no goals
                -/
          (by
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp ({ pt := Cat …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (fun x y =>  …
            -/
            ext1 ⟨i, j⟩
            /-
              case H.mk
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              i : Fin n
              j : (f i).ι
              ⊢ ∀ (j_1 : (CategoryTheory.Mat_.mk (Sigma fun j => (f j).ι) fun p => (f p.fst) …
            -/
            rintro ⟨i', j'⟩
            /-
              case H.mk.mk
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              i : Fin n
              j : (f i).ι
              i' : Fin n
              j' : (f i').ι
              ⊢ Eq (Finset.univ.sum (fun j => CategoryTheory.CategoryStruct.comp (fun x y => …
            -/
            rw [Finset.sum_apply, Finset.sum_apply]
            /-
              case H.mk.mk
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              i : Fin n
              j : (f i).ι
              i' : Fin n
              j' : (f i').ι
              ⊢ Eq (Finset.univ.sum fun c => CategoryTheory.CategoryStruct.comp (fun x y =>  …
            -/
            dsimp
            /-
              case H.mk.mk
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              i : Fin n
              j : (f i).ι
              i' : Fin n
              j' : (f i').ι
              ⊢ Eq (Finset.univ.sum fun c => Finset.univ.sum fun j_1 => CategoryTheory.Categ …
            -/
            rw [Finset.sum_eq_single i]; rotate_left
              /-
                case H.mk.mk.h₀
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                ⊢ ∀ (b : Fin n), Membership.mem Finset.univ b → Ne b i → Eq (Finset.univ.sum f …
              -/
            · intro b _ hb
              /-
                case H.mk.mk.h₀
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                b : Fin n
                a✝ : Membership.mem Finset.univ b
                hb : Ne b i
                ⊢ Eq (Finset.univ.sum fun j_1 => CategoryTheory.CategoryStruct.comp (dite (Eq  …
              -/
              apply Finset.sum_eq_zero
              /-
                case H.mk.mk.h₀.h
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                b : Fin n
                a✝ : Membership.mem Finset.univ b
                hb : Ne b i
                ⊢ ∀ (x : (f b).ι), Membership.mem Finset.univ x → Eq (CategoryTheory.CategoryS …
              -/
              intro x _
              /-
                case H.mk.mk.h₀.h
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                b : Fin n
                a✝¹ : Membership.mem Finset.univ b
                hb : Ne b i
                x : (f b).ι
                a✝ : Membership.mem Finset.univ x
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i b) (fun h => dite (Eq (Eq …
              -/
              rw [dif_neg hb.symm, zero_comp]
              /-
                🎉 no goals
              -/
              /-
                case H.mk.mk.h₁
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                ⊢ Not (Membership.mem Finset.univ i) → Eq (Finset.univ.sum fun j_1 => Category …
              -/
            · intro hi
              /-
                case H.mk.mk.h₁
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                hi : Not (Membership.mem Finset.univ i)
                ⊢ Eq (Finset.univ.sum fun j_1 => CategoryTheory.CategoryStruct.comp (dite (Eq  …
              -/
              simp at hi
              /-
                🎉 no goals
              -/
            /-
              case H.mk.mk
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              i : Fin n
              j : (f i).ι
              i' : Fin n
              j' : (f i').ι
              ⊢ Eq (Finset.univ.sum fun j_1 => CategoryTheory.CategoryStruct.comp (dite (Eq  …
            -/
            rw [Finset.sum_eq_single j]; rotate_left
              /-
                case H.mk.mk.h₀
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                ⊢ ∀ (b : (f i).ι), Membership.mem Finset.univ b → Ne b j → Eq (CategoryTheory. …
              -/
            · intro b _ hb
              /-
                case H.mk.mk.h₀
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                b : (f i).ι
                a✝ : Membership.mem Finset.univ b
                hb : Ne b j
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i i) (fun h => dite (Eq (Eq …
              -/
              rw [dif_pos rfl, dif_neg, zero_comp]
              /-
                case H.mk.mk.h₀.hnc
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                b : (f i).ι
                a✝ : Membership.mem Finset.univ b
                hb : Ne b j
                ⊢ Not (Eq (Eq.rec j ⋯) b)
              -/
              simp only
              /-
                case H.mk.mk.h₀.hnc
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                b : (f i).ι
                a✝ : Membership.mem Finset.univ b
                hb : Ne b j
                ⊢ Not (Eq j b)
              -/
              tauto
              /-
                🎉 no goals
              -/
              /-
                case H.mk.mk.h₁
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                ⊢ Not (Membership.mem Finset.univ j) → Eq (CategoryTheory.CategoryStruct.comp  …
              -/
            · intro hj
              /-
                case H.mk.mk.h₁
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                hj : Not (Membership.mem Finset.univ j)
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (dite (Eq i i) (fun h => dite (Eq (Eq …
              -/
              simp at hj
              /-
                🎉 no goals
              -/
            simp only [eqToHom_refl, dite_eq_ite, ite_true, Category.id_comp, ne_eq,
              Sigma.mk.inj_iff, not_and, id_def]
            /-
              case H.mk.mk
              C : Type u₁
              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
              inst✝ : CategoryTheory.Preadditive C
              n : Nat
              f : Fin n → CategoryTheory.Mat_ C
              i : Fin n
              j : (f i).ι
              i' : Fin n
              j' : (f i').ι
              ⊢ Eq (dite (Eq i' i) (fun h => dite (Eq (Eq.rec j' h) j) (fun h' => CategoryTh …
            -/
            by_cases h : i' = i
              /-
                case pos
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                h : Eq i' i
                ⊢ Eq (dite (Eq i' i) (fun h => dite (Eq (Eq.rec j' h) j) (fun h' => CategoryTh …
              -/
            · subst h
              /-
                case pos
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i' : Fin n
                j' j : (f i').ι
                ⊢ Eq (dite (Eq i' i') (fun h => dite (Eq (Eq.rec j' h) j) (fun h' => CategoryT …
              -/
              rw [dif_pos rfl]
              /-
                case pos
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i' : Fin n
                j' j : (f i').ι
                ⊢ Eq (dite (Eq (Eq.rec j' ⋯) j) (fun h' => CategoryTheory.eqToHom ⋯) fun h' => …
              -/
              simp only [heq_eq_eq, true_and]
              /-
                case pos
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i' : Fin n
                j' j : (f i').ι
                ⊢ Eq (dite (Eq j' j) (fun h' => CategoryTheory.eqToHom ⋯) fun h' => 0) (dite ( …
              -/
              by_cases h : j' = j
                /-
                  case pos
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  i' : Fin n
                  j' j : (f i').ι
                  h : Eq j' j
                  ⊢ Eq (dite (Eq j' j) (fun h' => CategoryTheory.eqToHom ⋯) fun h' => 0) (dite ( …
                -/
              · subst h
                /-
                  case pos
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  i' : Fin n
                  j' : (f i').ι
                  ⊢ Eq (dite (Eq j' j') (fun h' => CategoryTheory.eqToHom ⋯) fun h' => 0) (dite  …
                -/
                simp
                /-
                  🎉 no goals
                -/
                /-
                  case neg
                  C : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                  inst✝ : CategoryTheory.Preadditive C
                  n : Nat
                  f : Fin n → CategoryTheory.Mat_ C
                  i' : Fin n
                  j' j : (f i').ι
                  h : Not (Eq j' j)
                  ⊢ Eq (dite (Eq j' j) (fun h' => CategoryTheory.eqToHom ⋯) fun h' => 0) (dite ( …
                -/
              · rw [dif_neg h, dif_neg (Ne.symm h)]
                /-
                  🎉 no goals
                -/
              /-
                case neg
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                h : Not (Eq i' i)
                ⊢ Eq (dite (Eq i' i) (fun h => dite (Eq (Eq.rec j' h) j) (fun h' => CategoryTh …
              -/
            · rw [dif_neg h, dif_neg]
              /-
                case neg.hnc
                C : Type u₁
                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                inst✝ : CategoryTheory.Preadditive C
                n : Nat
                f : Fin n → CategoryTheory.Mat_ C
                i : Fin n
                j : (f i).ι
                i' : Fin n
                j' : (f i').ι
                h : Not (Eq i' i)
                ⊢ Not (And (Eq i i') (HEq j j'))
              -/
              tauto) }
              /-
                🎉 no goals
              -/


/-- A functor induces a functor of matrix categories.
-/
@[simps]
def mapMat_ (F : C ⥤ D) [Functor.Additive F] : Mat_ C ⥤ Mat_ D where
  obj M := ⟨M.ι, fun i => F.obj (M.X i)⟩
  map f i j := F.map (f i j)


/-- The identity functor induces the identity functor on matrix categories.
-/
@[simps!]
def mapMatId : (𝟭 C).mapMat_ ≅ 𝟭 (Mat_ C) :=
                                            /-
                                              C : Type u₁
                                              inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                              inst✝² : CategoryTheory.Preadditive C
                                              D : Type u_1
                                              inst✝¹ : CategoryTheory.Category.{v₁, u_1} D
                                              inst✝ : CategoryTheory.Preadditive D
                                              M : CategoryTheory.Mat_ C
                                              ⊢ Eq ((CategoryTheory.Functor.id C).mapMat_.obj M) ((CategoryTheory.Functor.id …
                                            -/
  NatIso.ofComponents (fun M => eqToIso (by cases M; rfl)) fun {M N} f => by
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{v₁, u_1} D
      inst✝ : CategoryTheory.Preadditive D
      M N : CategoryTheory.Mat_ C
      f : Quiver.Hom M N
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).mapMat …
    -/
    ext
    /-
      case H
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{v₁, u_1} D
      inst✝ : CategoryTheory.Preadditive D
      M N : CategoryTheory.Mat_ C
      f : Quiver.Hom M N
      i✝ : ((CategoryTheory.Functor.id C).mapMat_.obj M).ι
      j✝ : ((CategoryTheory.Functor.id (CategoryTheory.Mat_ C)).obj N).ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).mapMat …
    -/
    cases M; cases N
    /-
      case H.mk.mk
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      inst✝² : CategoryTheory.Preadditive C
      D : Type u_1
      inst✝¹ : CategoryTheory.Category.{v₁, u_1} D
      inst✝ : CategoryTheory.Preadditive D
      ι✝¹ : Type
      fintype✝¹ : Fintype ι✝¹
      X✝¹ : ι✝¹ → C
      i✝ : ((CategoryTheory.Functor.id C).mapMat_.obj (CategoryTheory.Mat_.mk ι✝¹ X✝ …
      ι✝ : Type
      fintype✝ : Fintype ι✝
      X✝ : ι✝ → C
      j✝ : ((CategoryTheory.Functor.id (CategoryTheory.Mat_ C)).obj (CategoryTheory. …
      f : Quiver.Hom (CategoryTheory.Mat_.mk ι✝¹ X✝¹) (CategoryTheory.Mat_.mk ι✝ X✝)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C).mapMat …
    -/
    simp [comp_dite, dite_comp]
    /-
      🎉 no goals
    -/


/-- Composite functors induce composite functors on matrix categories.
-/
@[simps!]
def mapMatComp {E : Type*} [Category.{v₁} E] [Preadditive E] (F : C ⥤ D) [Functor.Additive F]
    (G : D ⥤ E) [Functor.Additive G] : (F ⋙ G).mapMat_ ≅ F.mapMat_ ⋙ G.mapMat_ :=
                                            /-
                                              C : Type u₁
                                              inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
                                              inst✝⁶ : CategoryTheory.Preadditive C
                                              D : Type u_1
                                              inst✝⁵ : CategoryTheory.Category.{v₁, u_1} D
                                              inst✝⁴ : CategoryTheory.Preadditive D
                                              E : Type u_2
                                              inst✝³ : CategoryTheory.Category.{v₁, u_2} E
                                              inst✝² : CategoryTheory.Preadditive E
                                              F : CategoryTheory.Functor C D
                                              inst✝¹ : F.Additive
                                              G : CategoryTheory.Functor D E
                                              inst✝ : G.Additive
                                              M : CategoryTheory.Mat_ C
                                              ⊢ Eq ((F.comp G).mapMat_.obj M) ((F.mapMat_.comp G.mapMat_).obj M)
                                            -/
  NatIso.ofComponents (fun M => eqToIso (by cases M; rfl)) fun {M N} f => by
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.Preadditive C
      D : Type u_1
      inst✝⁵ : CategoryTheory.Category.{v₁, u_1} D
      inst✝⁴ : CategoryTheory.Preadditive D
      E : Type u_2
      inst✝³ : CategoryTheory.Category.{v₁, u_2} E
      inst✝² : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Additive
      G : CategoryTheory.Functor D E
      inst✝ : G.Additive
      M N : CategoryTheory.Mat_ C
      f : Quiver.Hom M N
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).mapMat_.map f) ((fun M => …
    -/
    ext
    /-
      case H
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.Preadditive C
      D : Type u_1
      inst✝⁵ : CategoryTheory.Category.{v₁, u_1} D
      inst✝⁴ : CategoryTheory.Preadditive D
      E : Type u_2
      inst✝³ : CategoryTheory.Category.{v₁, u_2} E
      inst✝² : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Additive
      G : CategoryTheory.Functor D E
      inst✝ : G.Additive
      M N : CategoryTheory.Mat_ C
      f : Quiver.Hom M N
      i✝ : ((F.comp G).mapMat_.obj M).ι
      j✝ : ((F.mapMat_.comp G.mapMat_).obj N).ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).mapMat_.map f) ((fun M => …
    -/
    cases M; cases N
    /-
      case H.mk.mk
      C : Type u₁
      inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁶ : CategoryTheory.Preadditive C
      D : Type u_1
      inst✝⁵ : CategoryTheory.Category.{v₁, u_1} D
      inst✝⁴ : CategoryTheory.Preadditive D
      E : Type u_2
      inst✝³ : CategoryTheory.Category.{v₁, u_2} E
      inst✝² : CategoryTheory.Preadditive E
      F : CategoryTheory.Functor C D
      inst✝¹ : F.Additive
      G : CategoryTheory.Functor D E
      inst✝ : G.Additive
      ι✝¹ : Type
      fintype✝¹ : Fintype ι✝¹
      X✝¹ : ι✝¹ → C
      i✝ : ((F.comp G).mapMat_.obj (CategoryTheory.Mat_.mk ι✝¹ X✝¹)).ι
      ι✝ : Type
      fintype✝ : Fintype ι✝
      X✝ : ι✝ → C
      j✝ : ((F.mapMat_.comp G.mapMat_).obj (CategoryTheory.Mat_.mk ι✝ X✝)).ι
      f : Quiver.Hom (CategoryTheory.Mat_.mk ι✝¹ X✝¹) (CategoryTheory.Mat_.mk ι✝ X✝)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.comp G).mapMat_.map f) ((fun M => …
    -/
    simp [comp_dite, dite_comp]
    /-
      🎉 no goals
    -/


/-- The embedding of `C` into `Mat_ C` as one-by-one matrices.
(We index the summands by `PUnit`.) -/
@[simps]
def embedding : C ⥤ Mat_ C where
  obj X := ⟨PUnit, fun _ => X⟩
  map f _ _ := f
                 /-
                   C : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                   inst✝ : CategoryTheory.Preadditive C
                   x✝ : C
                   ⊢ Eq ({ obj := fun X => CategoryTheory.Mat_.mk PUnit.{1} fun x => X, map := fu …
                 -/
  map_id _ := by ext ⟨⟩; simp
                         /-
                           🎉 no goals
                         -/
                     /-
                       C : Type u₁
                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                       inst✝ : CategoryTheory.Preadditive C
                       X✝ Y✝ Z✝ : C
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun X => CategoryTheory.Mat_.mk PUnit.{1} fun x => X, map := fu …
                     -/
  map_comp _ _ := by ext ⟨⟩; simp
                             /-
                               🎉 no goals
                             -/


instance : (embedding C).Faithful where
  map_injective h := congr_fun (congr_fun h PUnit.unit) PUnit.unit


instance : (embedding C).Full where map_surjective f := ⟨f PUnit.unit PUnit.unit, rfl⟩


instance : Functor.Additive (embedding C) where


instance [Inhabited C] : Inhabited (Mat_ C) :=
  ⟨(embedding C).obj default⟩


/-- Every object in `Mat_ C` is isomorphic to the biproduct of its summands.
-/
@[simps]
def isoBiproductEmbedding (M : Mat_ C) : M ≅ ⨁ fun i => (embedding C).obj (M.X i) where
  hom := biproduct.lift fun i j _ => if h : j = i then eqToHom (congr_arg M.X h) else 0
  inv := biproduct.desc fun i _ k => if h : i = k then eqToHom (congr_arg M.X h) else 0
  hom_inv_id := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.lift …
    -/
    simp only [biproduct.lift_desc]
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      ⊢ Eq (Finset.univ.sum fun j => CategoryTheory.CategoryStruct.comp (fun j_1 x = …
    -/
    funext i j
    /-
      case h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i j : M.ι
      ⊢ Eq (Finset.univ.sum (fun j => CategoryTheory.CategoryStruct.comp (fun j_1 x  …
    -/
    dsimp [id_def]
    /-
      case h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i j : M.ι
      ⊢ Eq (Finset.univ.sum (fun j => CategoryTheory.CategoryStruct.comp (fun j_1 x  …
    -/
    rw [Finset.sum_apply, Finset.sum_apply, Finset.sum_eq_single i]; rotate_left
      /-
        case h.h.h₀
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j : M.ι
        ⊢ ∀ (b : M.ι), Membership.mem Finset.univ b → Ne b i → Eq (CategoryTheory.Cate …
      -/
    · intro b _ hb
      /-
        case h.h.h₀
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j b : M.ι
        a✝ : Membership.mem Finset.univ b
        hb : Ne b i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun j x => dite (Eq j b) (fun h => C …
      -/
      dsimp
      /-
        case h.h.h₀
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j b : M.ι
        a✝ : Membership.mem Finset.univ b
        hb : Ne b i
        ⊢ Eq (Finset.univ.sum fun j_1 => CategoryTheory.CategoryStruct.comp (dite (Eq  …
      -/
      rw [Fintype.univ_ofSubsingleton, Finset.sum_singleton, dif_neg hb.symm, zero_comp]
      /-
        🎉 no goals
      -/
      /-
        case h.h.h₁
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j : M.ι
        ⊢ Not (Membership.mem Finset.univ i) → Eq (CategoryTheory.CategoryStruct.comp  …
      -/
    · intro h
      /-
        case h.h.h₁
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j : M.ι
        h : Not (Membership.mem Finset.univ i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun j x => dite (Eq j i) (fun h => C …
      -/
      simp at h
      /-
        🎉 no goals
      -/
    /-
      case h.h
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i j : M.ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun j x => dite (Eq j i) (fun h => C …
    -/
    simp
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.desc …
    -/
    apply biproduct.hom_ext
    /-
      case w
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      ⊢ ∀ (j : M.ι), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Category …
    -/
    intro i
    /-
      case w
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i : M.ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    apply biproduct.hom_ext'
    /-
      case w.w
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i : M.ι
      ⊢ ∀ (j : M.ι), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.b …
    -/
    intro j
    simp only [Category.id_comp, Category.assoc, biproduct.lift_π, biproduct.ι_desc_assoc,
      biproduct.ι_π]
    /-
      case w.w
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i j : M.ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x k => dite (Eq j k) (fun h => C …
    -/
    ext ⟨⟩ ⟨⟩
    simp only [embedding, comp_apply, comp_dite, dite_comp, comp_zero, zero_comp,
      Finset.sum_dite_eq', Finset.mem_univ, ite_true, eqToHom_refl, Category.comp_id]
    /-
      case w.w.H.unit.unit
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Preadditive C
      M : CategoryTheory.Mat_ C
      i j : M.ι
      ⊢ Eq (dite (Eq j i) (fun h => CategoryTheory.eqToHom ⋯) fun h => 0) (dite (Eq  …
    -/
    split_ifs with h
      /-
        case pos
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j : M.ι
        h : Eq j i
        ⊢ Eq (CategoryTheory.eqToHom ⋯) (CategoryTheory.eqToHom ⋯ PUnit.unit PUnit.unit)
      -/
    · subst h
      /-
        case pos
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        j : M.ι
        ⊢ Eq (CategoryTheory.eqToHom ⋯) (CategoryTheory.eqToHom ⋯ PUnit.unit PUnit.unit)
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u₁
        inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
        inst✝ : CategoryTheory.Preadditive C
        M : CategoryTheory.Mat_ C
        i j : M.ι
        h : Not (Eq j i)
        ⊢ Eq 0 (0 PUnit.unit PUnit.unit)
      -/
    · rfl
      /-
        🎉 no goals
      -/


instance (F : Mat_ C ⥤ D) [Functor.Additive F] (M : Mat_ C) :
    HasBiproduct (fun i => F.obj ((embedding C).obj (M.X i))) :=
  F.hasBiproduct_of_preserves _

-- Porting note: removed the @[simps] attribute as the automatically generated lemmas
-- are not very useful; two more useful lemmas have been added just after this
-- definition in order to ease the proof of `additiveObjIsoBiproduct_naturality`

/-- Every `M` is a direct sum of objects from `C`, and `F` preserves biproducts. -/
def additiveObjIsoBiproduct (F : Mat_ C ⥤ D) [Functor.Additive F] (M : Mat_ C) :
    F.obj M ≅ ⨁ fun i => F.obj ((embedding C).obj (M.X i)) :=
  F.mapIso (isoBiproductEmbedding M) ≪≫ F.mapBiproduct _


@[reassoc (attr := simp)]
lemma additiveObjIsoBiproduct_hom_π (F : Mat_ C ⥤ D) [Functor.Additive F] (M : Mat_ C) (i : M.ι) :
    (additiveObjIsoBiproduct F M).hom ≫ biproduct.π _ i =
      F.map (M.isoBiproductEmbedding.hom ≫ biproduct.π _ i) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M : CategoryTheory.Mat_ C
    i : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
  -/
  dsimp [additiveObjIsoBiproduct]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M : CategoryTheory.Mat_ C
    i : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [biproduct.lift_π, Category.assoc]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M : CategoryTheory.Mat_ C
    i : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.biprodu …
  -/
  erw [biproduct.lift_π, ← F.map_comp]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M : CategoryTheory.Mat_ C
    i : M.ι
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprodu …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_additiveObjIsoBiproduct_inv (F : Mat_ C ⥤ D) [Functor.Additive F] (M : Mat_ C) (i : M.ι) :
    biproduct.ι _ i ≫ (additiveObjIsoBiproduct F M).inv =
      F.map (biproduct.ι _ i ≫ M.isoBiproductEmbedding.inv) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M : CategoryTheory.Mat_ C
    i : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
  -/
  dsimp [additiveObjIsoBiproduct, Functor.mapBiproduct, Functor.mapBicone]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} D
    inst✝¹ : CategoryTheory.Preadditive D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M : CategoryTheory.Mat_ C
    i : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
  -/
  simp only [biproduct.ι_desc, biproduct.ι_desc_assoc, ← F.map_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem additiveObjIsoBiproduct_naturality (F : Mat_ C ⥤ D) [Functor.Additive F] {M N : Mat_ C}
    (f : M ⟶ N) :
    F.map f ≫ (additiveObjIsoBiproduct F N).hom =
      (additiveObjIsoBiproduct F M).hom ≫
        biproduct.matrix fun i j => F.map ((embedding C).map (f i j)) := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M N : CategoryTheory.Mat_ C
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.Mat_.additi …
  -/
  ext i : 1
  simp only [Category.assoc, additiveObjIsoBiproduct_hom_π, isoBiproductEmbedding_hom,
    embedding_obj_ι, embedding_obj_X, biproduct.lift_π, biproduct.matrix_π,
    ← cancel_epi (additiveObjIsoBiproduct F M).inv, Iso.inv_hom_id_assoc]
  /-
    case w
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M N : CategoryTheory.Mat_ C
    f : Quiver.Hom M N
    i : N.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
  -/
  ext j : 1
  simp only [ι_additiveObjIsoBiproduct_inv_assoc, isoBiproductEmbedding_inv,
    biproduct.ι_desc, ← F.map_comp]
  /-
    case w.w
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M N : CategoryTheory.Mat_ C
    f : Quiver.Hom M N
    i : N.ι
    j : M.ι
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (fun x k => dite (Eq j k) (fun …
  -/
  congr 1
  /-
    case w.w.e_a
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M N : CategoryTheory.Mat_ C
    f : Quiver.Hom M N
    i : N.ι
    j : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x k => dite (Eq j k) (fun h => C …
  -/
  funext ⟨⟩ ⟨⟩
  /-
    case w.w.e_a.h.h
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M N : CategoryTheory.Mat_ C
    f : Quiver.Hom M N
    i : N.ι
    j : M.ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x k => dite (Eq j k) (fun h => C …
  -/
  simp [comp_apply, dite_comp, comp_dite]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem additiveObjIsoBiproduct_naturality' (F : Mat_ C ⥤ D) [Functor.Additive F] {M N : Mat_ C}
    (f : M ⟶ N) :
    (additiveObjIsoBiproduct F M).inv ≫ F.map f =
      biproduct.matrix (fun i j => F.map ((embedding C).map (f i j)) : _) ≫
        (additiveObjIsoBiproduct F N).inv := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
    inst✝ : F.Additive
    M N : CategoryTheory.Mat_ C
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
  -/
  rw [Iso.inv_comp_eq, ← Category.assoc, Iso.eq_comp_inv, additiveObjIsoBiproduct_naturality]
  /-
    🎉 no goals
  -/


/-- Any additive functor `C ⥤ D` to a category `D` with finite biproducts extends to
a functor `Mat_ C ⥤ D`. -/
@[simps]
def lift (F : C ⥤ D) [Functor.Additive F] : Mat_ C ⥤ D where
  obj X := ⨁ fun i => F.obj (X.X i)
  map f := biproduct.matrix fun i j => F.map (f i j)
  map_id X := by
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Preadditive C
      D : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} D
      inst✝² : CategoryTheory.Preadditive D
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X : CategoryTheory.Mat_ C
      ⊢ Eq ({ obj := fun X => CategoryTheory.Limits.biproduct fun i => F.obj (X.X i) …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Preadditive C
      D : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} D
      inst✝² : CategoryTheory.Preadditive D
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X : CategoryTheory.Mat_ C
      ⊢ Eq (CategoryTheory.Limits.biproduct.matrix fun i j => F.map (CategoryTheory. …
    -/
    ext i j
    /-
      case w.w
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      inst✝⁴ : CategoryTheory.Preadditive C
      D : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} D
      inst✝² : CategoryTheory.Preadditive D
      inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
      F : CategoryTheory.Functor C D
      inst✝ : F.Additive
      X : CategoryTheory.Mat_ C
      i j : X.ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
    -/
    by_cases h : j = i
      /-
        case pos
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁴ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} D
        inst✝² : CategoryTheory.Preadditive D
        inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.Mat_ C
        i j : X.ι
        h : Eq j i
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
      -/
    · subst h; simp
               /-
                 🎉 no goals
               -/
      /-
        case neg
        C : Type u₁
        inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁴ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} D
        inst✝² : CategoryTheory.Preadditive D
        inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝ : F.Additive
        X : CategoryTheory.Mat_ C
        i j : X.ι
        h : Not (Eq j i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biproduct.ι (f …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/


instance lift_additive (F : C ⥤ D) [Functor.Additive F] : Functor.Additive (lift F) where


/-- An additive functor `C ⥤ D` factors through its lift to `Mat_ C ⥤ D`. -/
@[simps!]
def embeddingLiftIso (F : C ⥤ D) [Functor.Additive F] : embedding C ⋙ lift F ≅ F :=
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Preadditive C
    D : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : CategoryTheory.Limits.HasFiniteBiproducts D
    F : CategoryTheory.Functor C D
    inst✝ : F.Additive
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents
  /-
    🎉 no goals
  -/
    (fun X =>
      { hom := biproduct.desc fun _ => 𝟙 (F.obj X)
        inv := biproduct.lift fun _ => 𝟙 (F.obj X) })


/-- `Mat_.lift F` is the unique additive functor `L : Mat_ C ⥤ D` such that `F ≅ embedding C ⋙ L`.
-/
def liftUnique (F : C ⥤ D) [Functor.Additive F] (L : Mat_ C ⥤ D) [Functor.Additive L]
    (α : embedding C ⋙ L ≅ F) : L ≅ lift F :=
  NatIso.ofComponents
    (fun M =>
      additiveObjIsoBiproduct L M ≪≫
        (biproduct.mapIso fun i => α.app (M.X i)) ≪≫
          (biproduct.mapIso fun i => (embeddingLiftIso F).symm.app (M.X i)) ≪≫
            (additiveObjIsoBiproduct (lift F) M).symm)
    fun f => by
      /-
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map f) ((fun M => (CategoryTheory. …
      -/
      dsimp only [Iso.trans_hom, Iso.symm_hom, biproduct.mapIso_hom]
      /-
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map f) (CategoryTheory.CategoryStr …
      -/
      simp only [additiveObjIsoBiproduct_naturality_assoc]
      /-
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
      -/
      simp only [biproduct.matrix_map_assoc, Category.assoc]
      /-
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
      -/
      simp only [additiveObjIsoBiproduct_naturality']
      /-
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
      -/
      simp only [biproduct.map_matrix_assoc, Category.assoc]
      /-
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Mat_.additiveObjIsoBi …
      -/
      congr 3
      /-
        case e_a.e_a.e_m
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        ⊢ Eq (fun j k => CategoryTheory.CategoryStruct.comp (L.map ((CategoryTheory.Ma …
      -/
      ext j k
      /-
        case e_a.e_a.e_m.h.h
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        j : X✝.ι
        k : Y✝.ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map ((CategoryTheory.Mat_.embeddin …
      -/
      apply biproduct.hom_ext
      /-
        case e_a.e_a.e_m.h.h.w
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        j : X✝.ι
        k : Y✝.ι
        ⊢ ∀ (j_1 : ((CategoryTheory.Mat_.embedding C).obj (Y✝.X k)).ι), Eq (CategoryTh …
      -/
      rintro ⟨⟩
      /-
        case e_a.e_a.e_m.h.h.w.unit
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        j : X✝.ι
        k : Y✝.ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp
      /-
        case e_a.e_a.e_m.h.h.w.unit
        C : Type u₁
        inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
        inst✝⁵ : CategoryTheory.Preadditive C
        D : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
        inst✝³ : CategoryTheory.Preadditive D
        inst✝² : CategoryTheory.Limits.HasFiniteBiproducts D
        F : CategoryTheory.Functor C D
        inst✝¹ : F.Additive
        L : CategoryTheory.Functor (CategoryTheory.Mat_ C) D
        inst✝ : L.Additive
        α : CategoryTheory.Iso ((CategoryTheory.Mat_.embedding C).comp L) F
        X✝ Y✝ : CategoryTheory.Mat_ C
        f : Quiver.Hom X✝ Y✝
        j : X✝.ι
        k : Y✝.ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simpa using α.hom.naturality (f j k)
      /-
        🎉 no goals
      -/

-- TODO is there some uniqueness statement for the natural isomorphism in `liftUnique`?

/-- Two additive functors `Mat_ C ⥤ D` are naturally isomorphic if
their precompositions with `embedding C` are naturally isomorphic as functors `C ⥤ D`. -/
def ext {F G : Mat_ C ⥤ D} [Functor.Additive F] [Functor.Additive G]
    (α : embedding C ⋙ F ≅ embedding C ⋙ G) : F ≅ G :=
  liftUnique (embedding C ⋙ G) _ α ≪≫ (liftUnique _ _ (Iso.refl _)).symm


/-- Natural isomorphism needed in the construction of `equivalenceSelfOfHasFiniteBiproducts`.
-/
def equivalenceSelfOfHasFiniteBiproductsAux [HasFiniteBiproducts C] :
    embedding C ⋙ 𝟭 (Mat_ C) ≅ embedding C ⋙ lift (𝟭 C) ⋙ embedding C :=
  Functor.rightUnitor _ ≪≫
    (Functor.leftUnitor _).symm ≪≫
      isoWhiskerRight (embeddingLiftIso _).symm _ ≪≫ Functor.associator _ _ _


/--
A preadditive category that already has finite biproducts is equivalent to its additive envelope.

Note that we only prove this for a large category;
otherwise there are universe issues that I haven't attempted to sort out.
-/
def equivalenceSelfOfHasFiniteBiproducts (C : Type (u₁ + 1)) [LargeCategory C] [Preadditive C]
    [HasFiniteBiproducts C] : Mat_ C ≌ C :=
  Equivalence.mk
    (-- I suspect this is already an adjoint equivalence, but it seems painful to verify.
      lift
      (𝟭 C))
    (embedding C) (ext equivalenceSelfOfHasFiniteBiproductsAux) (embeddingLiftIso (𝟭 C))


@[simp]
theorem equivalenceSelfOfHasFiniteBiproducts_functor {C : Type (u₁ + 1)} [LargeCategory C]
    [Preadditive C] [HasFiniteBiproducts C] :
    (equivalenceSelfOfHasFiniteBiproducts C).functor = lift (𝟭 C) :=
  rfl


@[simp]
theorem equivalenceSelfOfHasFiniteBiproducts_inverse {C : Type (u₁ + 1)} [LargeCategory C]
    [Preadditive C] [HasFiniteBiproducts C] :
    (equivalenceSelfOfHasFiniteBiproducts C).inverse = embedding C :=
  rfl


/-- A type synonym for `Fintype`, which we will equip with a category structure
where the morphisms are matrices with components in `R`. -/
@[nolint unusedArguments]
def Mat (_ : Type u) :=
  FintypeCat.{u}


instance (R : Type u) : Inhabited (Mat R) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    R : Type u
    ⊢ Inhabited (CategoryTheory.Mat R)
  -/
  dsimp [Mat]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Preadditive C
    R : Type u
    ⊢ Inhabited FintypeCat
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (R : Type u) : CoeSort (Mat R) (Type u) :=
  Bundled.coeSort


instance (R : Type u) [Semiring R] : Category (Mat R) where
  Hom X Y := Matrix X Y R
  id X := (1 : Matrix X X R)
  comp {X Y Z} f g := (show Matrix X Y R from f) * (show Matrix Y Z R from g)
              /-
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                inst✝¹ : CategoryTheory.Preadditive C
                R : Type u
                inst✝ : Semiring R
                ⊢ ∀ {W X Y Z : CategoryTheory.Mat R} (f : Quiver.Hom W X) (g : Quiver.Hom X Y) …
              -/
  assoc := by intros; simp [Matrix.mul_assoc]
                      /-
                        🎉 no goals
                      -/


@[ext]
theorem hom_ext {X Y : Mat R} (f g : X ⟶ Y) (h : ∀ i j, f i j = g i j) : f = g :=
  Matrix.ext_iff.mp h


theorem id_def (M : Mat R) : 𝟙 M = fun i j => if i = j then 1 else 0 :=
  rfl


theorem id_apply (M : Mat R) (i j : M) : (𝟙 M : Matrix M M R) i j = if i = j then 1 else 0 :=
  rfl


@[simp]
                                                                               /-
                                                                                 R : Type u
                                                                                 inst✝ : Semiring R
                                                                                 M : CategoryTheory.Mat R
                                                                                 i : ↑M
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.id M i i) 1
                                                                               -/
theorem id_apply_self (M : Mat R) (i : M) : (𝟙 M : Matrix M M R) i i = 1 := by simp [id_apply]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[simp]
theorem id_apply_of_ne (M : Mat R) (i j : M) (h : i ≠ j) : (𝟙 M : Matrix M M R) i j = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    M : CategoryTheory.Mat R
    i j : ↑M
    h : Ne i j
    ⊢ Eq (CategoryTheory.CategoryStruct.id M i j) 0
  -/
  simp [id_apply, h]
  /-
    🎉 no goals
  -/


theorem comp_def {M N K : Mat R} (f : M ⟶ N) (g : N ⟶ K) :
    f ≫ g = fun i k => ∑ j : N, f i j * g j k :=
  rfl


@[simp]
theorem comp_apply {M N K : Mat R} (f : M ⟶ N) (g : N ⟶ K) (i k) :
    (f ≫ g) i k = ∑ j : N, f i j * g j k :=
  rfl


instance (M N : Mat R) : Inhabited (M ⟶ N) :=
  ⟨fun (_ : M) (_ : N) => (0 : R)⟩


/-- Auxiliary definition for `CategoryTheory.Mat.equivalenceSingleObj`. -/
@[simps]
def equivalenceSingleObjInverse : Mat_ (SingleObj Rᵐᵒᵖ) ⥤ Mat R where
  obj X := FintypeCat.of X.ι
  map f i j := MulOpposite.unop (f i j)
  map_id X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      ⊢ Eq ({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOpposi …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      i✝ j✝ : ↑({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOp …
      ⊢ Eq ({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOpposi …
    -/
    simp only [Mat_.id_def, id_def]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      i✝ j✝ : ↑({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOp …
      ⊢ Eq (MulOpposite.unop (dite (Eq i✝ j✝) (fun h => CategoryTheory.eqToHom ⋯) fu …
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> rfl
                  /-
                    🎉 no goals
                  -/
  map_comp f g := by
    -- Porting note: this proof was automatic in mathlib3
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X✝ Y✝ Z✝ : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOpposi …
    -/
    ext
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X✝ Y✝ Z✝ : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      i✝ : ↑({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOppos …
      j✝ : ↑({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOppos …
      ⊢ Eq ({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOpposi …
    -/
    simp only [Mat_.comp_apply, comp_apply]
    /-
      case h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X✝ Y✝ Z✝ : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      i✝ : ↑({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOppos …
      j✝ : ↑({ obj := fun X => FintypeCat.of X.ι, map := fun {X Y} f i j => MulOppos …
      ⊢ Eq (MulOpposite.unop (Finset.univ.sum fun j => CategoryTheory.CategoryStruct …
    -/
    apply Finset.unop_sum
    /-
      🎉 no goals
    -/


instance : (equivalenceSingleObjInverse R).Faithful where
  map_injective w := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X✝ Y✝ : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      w : Eq ((CategoryTheory.Mat.equivalenceSingleObjInverse R).map a₁✝) ((Category …
      ⊢ Eq a₁✝ a₂✝
    -/
    ext
    /-
      case H
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X✝ Y✝ : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      w : Eq ((CategoryTheory.Mat.equivalenceSingleObjInverse R).map a₁✝) ((Category …
      i✝ : X✝.ι
      j✝ : Y✝.ι
      ⊢ Eq (a₁✝ i✝ j✝) (a₂✝ i✝ j✝)
    -/
    apply_fun MulOpposite.unop using MulOpposite.unop_injective
    /-
      case H
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      inst✝¹ : CategoryTheory.Preadditive C
      R : Type
      inst✝ : Ring R
      X✝ Y✝ : CategoryTheory.Mat_ (CategoryTheory.SingleObj (MulOpposite R))
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      w : Eq ((CategoryTheory.Mat.equivalenceSingleObjInverse R).map a₁✝) ((Category …
      i✝ : X✝.ι
      j✝ : Y✝.ι
      ⊢ Eq (MulOpposite.unop (a₁✝ i✝ j✝)) (MulOpposite.unop (a₂✝ i✝ j✝))
    -/
    exact congr_fun (congr_fun w _) _
    /-
      🎉 no goals
    -/


instance : (equivalenceSingleObjInverse R).Full where
  map_surjective f := ⟨fun i j => MulOpposite.op (f i j), rfl⟩


instance : (equivalenceSingleObjInverse R).EssSurj where
  mem_essImage X :=
    ⟨{  ι := X
                                                 /-
                                                   C : Type u₁
                                                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                   inst✝¹ : CategoryTheory.Preadditive C
                                                   R : Type
                                                   inst✝ : Ring R
                                                   X : CategoryTheory.Mat R
                                                   ⊢ Eq ((CategoryTheory.Mat.equivalenceSingleObjInverse R).obj (CategoryTheory.M …
                                                 -/
        X := fun _ => PUnit.unit }, ⟨eqToIso (by dsimp; cases X; congr)⟩⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance : (equivalenceSingleObjInverse R).IsEquivalence where


/-- The categorical equivalence between the category of matrices over a ring,
and the category of matrices over that ring considered as a single-object category. -/
def equivalenceSingleObj : Mat R ≌ Mat_ (SingleObj Rᵐᵒᵖ) :=
  (equivalenceSingleObjInverse R).asEquivalence.symm

-- Porting note: added as this was not found automatically

instance (X Y : Mat R) : AddCommGroup (X ⟶ Y) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Preadditive C
    R : Type
    inst✝ : Ring R
    X Y : CategoryTheory.Mat R
    ⊢ AddCommGroup (Quiver.Hom X Y)
  -/
  change AddCommGroup (Matrix X Y R)
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    inst✝¹ : CategoryTheory.Preadditive C
    R : Type
    inst✝ : Ring R
    X Y : CategoryTheory.Mat R
    ⊢ AddCommGroup (Matrix (↑X) (↑Y) R)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem add_apply {M N : Mat R} (f g : M ⟶ N) (i j) : (f + g) i j = f i j + g i j :=
  rfl


instance : Preadditive (Mat R) where

-- TODO show `Mat R` has biproducts, and that `biprod.map` "is" forming a block diagonal matrix.

