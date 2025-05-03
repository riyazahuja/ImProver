/-- Given two composable morphisms `f : X₁ ⟶ X₂` and `g : X₂ ⟶ X₃` in the category
of cochain complexes, this is the canonical triangle
`mappingCone f ⟶ mappingCone (f ≫ g) ⟶ mappingCone g ⟶ (mappingCone f)⟦1⟧`. -/
@[simps! mor₁ mor₂ mor₃ obj₁ obj₂ obj₃]
noncomputable def mappingConeCompTriangle : Triangle (CochainComplex C ℤ) :=
                                          /-
                                            C : Type u_1
                                            inst✝² : CategoryTheory.Category.{?u.326, u_1} C
                                            inst✝¹ : CategoryTheory.Preadditive C
                                            inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                            X₁ X₂ X₃ : CochainComplex C Int
                                            f : Quiver.Hom X₁ X₂
                                            g : Quiver.Hom X₂ X₃
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
                                          -/
  Triangle.mk (map f (f ≫ g) (𝟙 X₁) g (by rw [id_comp]))
                                          /-
                                            🎉 no goals
                                          -/
                                /-
                                  C : Type u_1
                                  inst✝² : CategoryTheory.Category.{?u.326, u_1} C
                                  inst✝¹ : CategoryTheory.Preadditive C
                                  inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                  X₁ X₂ X₃ : CochainComplex C Int
                                  f : Quiver.Hom X₁ X₂
                                  g : Quiver.Hom X₂ X₃
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                -/
    (map (f ≫ g) g f (𝟙 X₃) (by rw [comp_id]))
                                /-
                                  🎉 no goals
                                -/
    ((triangle g).mor₃ ≫ (inr f)⟦1⟧')


/-- Given two composable morphisms `f : X₁ ⟶ X₂` and `g : X₂ ⟶ X₃` in the category
of cochain complexes, this is the canonical triangle
`mappingCone f ⟶ mappingCone (f ≫ g) ⟶ mappingCone g ⟶ (mappingCone f)⟦1⟧`
in the homotopy category. It is a distinguished triangle,
see `HomotopyCategory.mappingConeCompTriangleh_distinguished`. -/
noncomputable def mappingConeCompTriangleh :
    Triangle (HomotopyCategory C (ComplexShape.up ℤ)) :=
  (HomotopyCategory.quotient _ _).mapTriangle.obj (mappingConeCompTriangle f g)


@[reassoc]
lemma mappingConeCompTriangle_mor₃_naturality {Y₁ Y₂ Y₃ : CochainComplex C ℤ} (f' : Y₁ ⟶ Y₂)
    (g' : Y₂ ⟶ Y₃) (φ : mk₂ f g ⟶ mk₂ f' g') :
                                  /-
                                    C : Type u_1
                                    inst✝² : CategoryTheory.Category.{?u.10275, u_1} C
                                    inst✝¹ : CategoryTheory.Preadditive C
                                    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                    X₁ X₂ X₃ : CochainComplex C Int
                                    f : Quiver.Hom X₁ X₂
                                    g : Quiver.Hom X₂ X₃
                                    Y₁ Y₂ Y₃ : CochainComplex C Int
                                    f' : Quiver.Hom Y₁ Y₂
                                    g' : Quiver.Hom Y₂ Y₃
                                    φ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₂ f g) (CategoryTheory.Compo …
                                    ⊢ LE.le 1 2
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
    map g g' (φ.app 1) (φ.app 2) (naturality' φ 1 2) ≫ (mappingConeCompTriangle f' g').mor₃ =
                                  /-
                                    🎉 no goals
                                  -/
      (mappingConeCompTriangle f g).mor₃ ≫
                                       /-
                                         C : Type u_1
                                         inst✝² : CategoryTheory.Category.{?u.10275, u_1} C
                                         inst✝¹ : CategoryTheory.Preadditive C
                                         inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                         X₁ X₂ X₃ : CochainComplex C Int
                                         f : Quiver.Hom X₁ X₂
                                         g : Quiver.Hom X₂ X₃
                                         Y₁ Y₂ Y₃ : CochainComplex C Int
                                         f' : Quiver.Hom Y₁ Y₂
                                         g' : Quiver.Hom Y₂ Y₃
                                         φ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₂ f g) (CategoryTheory.Compo …
                                         ⊢ LE.le 0 1
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
        (map f f' (φ.app 0) (φ.app 1) (naturality' φ 0 1))⟦1⟧' := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    Y₁ Y₂ Y₃ : CochainComplex C Int
    f' : Quiver.Hom Y₁ Y₂
    g' : Quiver.Hom Y₂ Y₃
    φ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₂ f g) (CategoryTheory.Compo …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.map g g'  …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    Y₁ Y₂ Y₃ : CochainComplex C Int
    f' : Quiver.Hom Y₁ Y₂
    g' : Quiver.Hom Y₂ Y₃
    φ : Quiver.Hom (CategoryTheory.ComposableArrows.mk₂ f g) (CategoryTheory.Compo …
    n : Int
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.map g g' …
  -/
  dsimp [map]
  -- the following list of lemmas was obtained by doing simp? [ext_from_iff _ (n + 1) _ rfl]
  simp only [Int.reduceNeg, Fin.isValue, assoc, inr_f_desc_f, HomologicalComplex.comp_f,
    ext_from_iff _ (n + 1) _ rfl, inl_v_desc_f_assoc, Cochain.zero_cochain_comp_v, Cochain.ofHom_v,
    inl_v_triangle_mor₃_f_assoc, triangle_obj₁, shiftFunctor_obj_X', shiftFunctor_obj_X,
    shiftFunctorObjXIso, HomologicalComplex.XIsoOfEq_rfl, Iso.refl_inv, Preadditive.neg_comp,
    id_comp, Preadditive.comp_neg, inr_f_desc_f_assoc, inr_f_triangle_mor₃_f_assoc, zero_comp,
    comp_zero, and_self]


/-- Given two composable morphisms `f` and `g` in the category of cochain complexes, this
is the canonical morphism (which is an homotopy equivalence) from `mappingCone g` to
the mapping cone of the morphism `mappingCone f ⟶ mappingCone (f ≫ g)`. -/
noncomputable def hom :
    mappingCone g ⟶ mappingCone (mappingConeCompTriangle f g).mor₁ :=
                                                                   /-
                                                                     C : Type u_1
                                                                     inst✝² : CategoryTheory.Category.{?u.25144, u_1} C
                                                                     inst✝¹ : CategoryTheory.Preadditive C
                                                                     inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                                                     X₁ X₂ X₃ : CochainComplex C Int
                                                                     f : Quiver.Hom X₁ X₂
                                                                     g : Quiver.Hom X₂ X₃
                                                                     ⊢ Eq (CochainComplex.HomComplex.δ 0 1 (CochainComplex.HomComplex.Cochain.ofHom …
                                                                   -/
  lift _ (descCocycle g (Cochain.ofHom (inr f)) 0 (zero_add 1) (by dsimp; simp))
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    (descCochain _ 0 (Cochain.ofHom (inr (f ≫ g))) (neg_add_cancel 1)) (by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.25144, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        ⊢ Eq (HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 (CochainComplex.mappingCone.d …
      -/
      ext p _ rfl
      /-
        case h
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.25144, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        p : Int
        ⊢ Eq ((HAdd.hAdd (CochainComplex.HomComplex.δ 0 1 (CochainComplex.mappingCone. …
      -/
      dsimp [mappingConeCompTriangle, map]
      /-
        case h
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.25144, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        p : Int
        ⊢ Eq (HAdd.hAdd ((CochainComplex.HomComplex.δ 0 1 (CochainComplex.mappingCone. …
      -/
      simp [ext_from_iff _ _ _ rfl, inl_v_d_assoc _ (p+1) p (p+2) (by omega) (by omega)])
      /-
        🎉 no goals
      -/


/-- Given two composable morphisms `f` and `g` in the category of cochain complexes, this
is the canonical morphism (which is an homotopy equivalence) from the mapping cone of
the morphism `mappingCone f ⟶ mappingCone (f ≫ g)` to `mappingCone g`. -/
noncomputable def inv : mappingCone (mappingConeCompTriangle f g).mor₁ ⟶ mappingCone g :=
  desc _ ((snd f).comp (inl g) (zero_add (-1)))
                                                                         /-
                                                                           C : Type u_1
                                                                           inst✝² : CategoryTheory.Category.{?u.64515, u_1} C
                                                                           inst✝¹ : CategoryTheory.Preadditive C
                                                                           inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                                                           X₁ X₂ X₃ : CochainComplex C Int
                                                                           f : Quiver.Hom X₁ X₂
                                                                           g : Quiver.Hom X₂ X₃
                                                                           ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 ((CochainComplex.HomComplex.Cochain.o …
                                                                         -/
    (desc _ ((Cochain.ofHom f).comp (inl g) (zero_add (-1))) (inr g) (by simp)) (by
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.64515, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 ((CochainComplex.mappingCone.snd f).c …
      -/
      ext p
      /-
        case h
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.64515, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        p : Int
        ⊢ Eq ((CochainComplex.HomComplex.δ (-1) 0 ((CochainComplex.mappingCone.snd f). …
      -/
      rw [ext_from_iff _ (p + 1) _ rfl, ext_to_iff _ _ (p + 1) rfl]
      simp [map, δ_zero_cochain_comp,
        Cochain.comp_v _ _ (add_neg_cancel 1) p (p+1) p (by omega) (by omega)])

@[reassoc (attr := simp)]
lemma hom_inv_id : hom f g ≫ inv f g = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.MappingConeCompHomoto …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    n : Int
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CochainComplex.MappingConeCompHomot …
  -/
  simp [hom, inv, lift_desc_f _ _ _ _ _ _ _ n (n+1) rfl, ext_from_iff _ (n + 1) _ rfl]
  /-
    🎉 no goals
  -/


set_option maxHeartbeats 400000 in
/-- Given two composable morphisms `f` and `g` in the category of cochain complexes,
this is the `homotopyInvHomId` field of the homotopy equivalence
`mappingConeCompHomotopyEquiv f g` between `mappingCone g` and the mapping cone of
the morphism `mappingCone f ⟶ mappingCone (f ≫ g)`. -/
noncomputable def homotopyInvHomId : Homotopy (inv f g ≫ hom f g) (𝟙 _) :=
  (Cochain.equivHomotopy _ _).symm ⟨-((snd _).comp ((fst (f ≫ g)).1.comp
                              /-
                                C : Type u_1
                                inst✝² : CategoryTheory.Category.{?u.108690, u_1} C
                                inst✝¹ : CategoryTheory.Preadditive C
                                inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                X₁ X₂ X₃ : CochainComplex C Int
                                f : Quiver.Hom X₁ X₂
                                g : Quiver.Hom X₂ X₃
                                ⊢ Eq (HAdd.hAdd (-1) (-1)) (-2)
                              -/
                              /-
                                🎉 no goals
                              -/
    ((inl f).comp (inl _) (by omega)) (show 1 + (-2) = -1 by omega)) (zero_add (-1))), by
                                                             /-
                                                               🎉 no goals
                                                             -/
      rw [δ_neg, δ_zero_cochain_comp _ _ _ (neg_add_cancel 1),
        Int.negOnePow_neg, Int.negOnePow_one, Units.neg_smul, one_smul,
        δ_comp _ _ (show 1 + (-2) = -1 by omega) 2 (-1) 0 (by omega)
          (by omega) (by omega),
        δ_comp _ _ (show (-1) + (-1) = -2 by omega) 0 0 (-1) (by omega)
          (by omega) (by omega), Int.negOnePow_neg, Int.negOnePow_neg,
        Int.negOnePow_even 2 ⟨1, by omega⟩, Int.negOnePow_one, Units.neg_smul,
        one_smul, one_smul, δ_inl, δ_inl, δ_snd, Cocycle.δ_eq_zero, Cochain.zero_comp, add_zero,
        Cochain.neg_comp, neg_neg]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.108690, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        ⊢ Eq (CochainComplex.HomComplex.Cochain.ofHom (CategoryTheory.CategoryStruct.c …
      -/
      ext n
      rw [ext_from_iff _ (n + 1) n rfl, ext_from_iff _ (n + 1) n rfl,
        ext_from_iff _ (n + 2) (n + 1) (by omega)]
      /-
        case h
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.108690, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁ X₂ X₃ : CochainComplex C Int
        f : Quiver.Hom X₁ X₂
        g : Quiver.Hom X₂ X₃
        n : Int
        ⊢ And (And (Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCon …
      -/
      dsimp [hom, inv]
      simp [ext_to_iff _ n (n + 1) rfl, map, Cochain.comp_v _ _
          (add_neg_cancel 1) n (n + 1) n (by omega) (by omega),
        Cochain.comp_v _ _ (show 1 + -2 = -1 by omega) (n + 1) (n + 2) n
          (by omega) (by omega),
        Cochain.comp_v _ _ (show (-1) + -1 = -2 by omega) (n + 2) (n + 1) n
          (by omega) (by omega)]⟩


/-- Given two composable morphisms `f` and `g` in the category of cochain complexes,
this is the homotopy equivalence `mappingConeCompHomotopyEquiv f g`
between `mappingCone g` and the mapping cone of
the morphism `mappingCone f ⟶ mappingCone (f ≫ g)`. -/
noncomputable def mappingConeCompHomotopyEquiv : HomotopyEquiv (mappingCone g)
    (mappingCone (mappingConeCompTriangle f g).mor₁) where
  hom := MappingConeCompHomotopyEquiv.hom f g
  inv := MappingConeCompHomotopyEquiv.inv f g
                                        /-
                                          C : Type u_1
                                          inst✝² : CategoryTheory.Category.{?u.241010, u_1} C
                                          inst✝¹ : CategoryTheory.Preadditive C
                                          inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                          X₁ X₂ X₃ : CochainComplex C Int
                                          f : Quiver.Hom X₁ X₂
                                          g : Quiver.Hom X₂ X₃
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.MappingConeCompHomoto …
                                        -/
  homotopyHomInvId := Homotopy.ofEq (by simp)
                                        /-
                                          🎉 no goals
                                        -/
  homotopyInvHomId := MappingConeCompHomotopyEquiv.homotopyInvHomId f g


@[reassoc (attr := simp)]
lemma mappingConeCompHomotopyEquiv_hom_inv_id :
    (mappingConeCompHomotopyEquiv f g).hom ≫
      (mappingConeCompHomotopyEquiv f g).inv = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingConeCompHomoto …
  -/
  simp [mappingConeCompHomotopyEquiv]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mappingConeCompHomotopyEquiv_comm₁ :
                                    /-
                                      C : Type u_1
                                      inst✝² : CategoryTheory.Category.{?u.244829, u_1} C
                                      inst✝¹ : CategoryTheory.Preadditive C
                                      inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
                                      X₁ X₂ X₃ : CochainComplex C Int
                                      f : Quiver.Hom X₁ X₂
                                      g : Quiver.Hom X₂ X₃
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
                                    -/
    inr (map f (f ≫ g) (𝟙 X₁) g (by rw [id_comp])) ≫
                                    /-
                                      🎉 no goals
                                    -/
      (mappingConeCompHomotopyEquiv f g).inv = (mappingConeCompTriangle f g).mor₂ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.inr (Coch …
  -/
  simp [map, mappingConeCompHomotopyEquiv, MappingConeCompHomotopyEquiv.inv]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma mappingConeCompHomotopyEquiv_comm₂ :
    (mappingConeCompHomotopyEquiv f g).hom ≫
      (triangle (mappingConeCompTriangle f g).mor₁).mor₃ =
      (mappingConeCompTriangle f g).mor₃ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingConeCompHomoto …
  -/
  ext n
  simp [map, mappingConeCompHomotopyEquiv, MappingConeCompHomotopyEquiv.hom,
    lift_f _ _ _ _ _ (n+1) rfl, ext_from_iff _ (n+1) _ rfl]


@[reassoc (attr := simp)]
lemma mappingConeCompTriangleh_comm₁ :
    (mappingConeCompTriangleh f g).mor₂ ≫
      (HomotopyCategory.quotient _ _).map (mappingConeCompHomotopyEquiv f g).hom =
    (HomotopyCategory.quotient _ _).map (mappingCone.inr _) := by
  rw [← cancel_mono (HomotopyCategory.isoOfHomotopyEquiv
      (mappingConeCompHomotopyEquiv f g)).inv, assoc]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingConeCompTriang …
  -/
  dsimp [mappingConeCompTriangleh]
  rw [← Functor.map_comp, ← Functor.map_comp, ← Functor.map_comp,
    mappingConeCompHomotopyEquiv_hom_inv_id, comp_id,
    mappingConeCompHomotopyEquiv_comm₁ f g,
    mappingConeCompTriangle_mor₂]


lemma mappingConeCompTriangleh_distinguished :
    (mappingConeCompTriangleh f g) ∈
      distTriang (HomotopyCategory C (ComplexShape.up ℤ)) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cochai …
  -/
  refine ⟨_, _, (mappingConeCompTriangle f g).mor₁, ⟨?_⟩⟩
  refine Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (isoOfHomotopyEquiv
    (mappingConeCompHomotopyEquiv f g)) (by aesop_cat) (by simp) ?_
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingConeCompTriang …
  -/
  dsimp [mappingConeCompTriangleh]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [CategoryTheory.Functor.map_id, comp_id, ← Functor.map_comp_assoc]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient C (Comple …
  -/
  congr 2
  /-
    case e_a.e_a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
    X₁ X₂ X₃ : CochainComplex C Int
    f : Quiver.Hom X₁ X₂
    g : Quiver.Hom X₂ X₃
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.triangle  …
  -/
  exact (mappingConeCompHomotopyEquiv_comm₂ f g).symm
  /-
    🎉 no goals
  -/


noncomputable instance : IsTriangulated (HomotopyCategory C (ComplexShape.up ℤ)) :=
  IsTriangulated.mk' (by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁ X₂ X₃ : CochainComplex C Int
      f : Quiver.Hom X₁ X₂
      g : Quiver.Hom X₂ X₃
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      ⊢ ∀ ⦃X₁' X₂' X₃' : HomotopyCategory C (ComplexShape.up Int)⦄ (u₁₂' : Quiver.Ho …
    -/
    rintro ⟨X₁ : CochainComplex C ℤ⟩ ⟨X₂ : CochainComplex C ℤ⟩ ⟨X₃ : CochainComplex C ℤ⟩ u₁₂' u₂₃'
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁✝ X₂✝ X₃✝ : CochainComplex C Int
      f : Quiver.Hom X₁✝ X₂✝
      g : Quiver.Hom X₂✝ X₃✝
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X₁ X₂ X₃ : CochainComplex C Int
      u₁₂' : Quiver.Hom { as := X₁ } { as := X₂ }
      u₂₃' : Quiver.Hom { as := X₂ } { as := X₃ }
      ⊢ Exists fun X₁_1 => Exists fun X₂_1 => Exists fun X₃_1 => Exists fun Z₁₂ => E …
    -/
    obtain ⟨u₁₂, rfl⟩ := (HomotopyCategory.quotient C (ComplexShape.up ℤ)).map_surjective u₁₂'
    /-
      case mk.mk.mk.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁✝ X₂✝ X₃✝ : CochainComplex C Int
      f : Quiver.Hom X₁✝ X₂✝
      g : Quiver.Hom X₂✝ X₃✝
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X₁ X₂ X₃ : CochainComplex C Int
      u₂₃' : Quiver.Hom { as := X₂ } { as := X₃ }
      u₁₂ : Quiver.Hom X₁ X₂
      ⊢ Exists fun X₁_1 => Exists fun X₂_1 => Exists fun X₃_1 => Exists fun Z₁₂ => E …
    -/
    obtain ⟨u₂₃, rfl⟩ := (HomotopyCategory.quotient C (ComplexShape.up ℤ)).map_surjective u₂₃'
    refine ⟨_, _, _, _, _, _, _, _, Iso.refl _, Iso.refl _, Iso.refl _, by simp, by simp,
        _, _, mappingCone_triangleh_distinguished u₁₂,
        _, _, mappingCone_triangleh_distinguished u₂₃,
        _, _, mappingCone_triangleh_distinguished (u₁₂ ≫ u₂₃), ⟨?_⟩⟩
    /-
      case mk.mk.mk.intro.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁✝ X₂✝ X₃✝ : CochainComplex C Int
      f : Quiver.Hom X₁✝ X₂✝
      g : Quiver.Hom X₂✝ X₃✝
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X₁ X₂ X₃ : CochainComplex C Int
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      ⊢ CategoryTheory.Triangulated.Octahedron ⋯ ⋯ ⋯ ⋯
    -/
    let α := mappingCone.triangleMap u₁₂ (u₁₂ ≫ u₂₃) (𝟙 X₁) u₂₃ (by rw [id_comp])
    /-
      case mk.mk.mk.intro.intro
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{u_2, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
      X₁✝ X₂✝ X₃✝ : CochainComplex C Int
      f : Quiver.Hom X₁✝ X₂✝
      g : Quiver.Hom X₂✝ X₃✝
      inst✝ : CategoryTheory.Limits.HasZeroObject C
      X₁ X₂ X₃ : CochainComplex C Int
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      α : Quiver.Hom (CochainComplex.mappingCone.triangle u₁₂) (CochainComplex.mappi …
      ⊢ CategoryTheory.Triangulated.Octahedron ⋯ ⋯ ⋯ ⋯
    -/
    let β := mappingCone.triangleMap (u₁₂ ≫ u₂₃) u₂₃ u₁₂ (𝟙 X₃) (by rw [comp_id])
    refine Triangulated.Octahedron.mk ((HomotopyCategory.quotient _ _).map α.hom₃)
      ((HomotopyCategory.quotient _ _).map β.hom₃) ?_ ?_ ?_ ?_ ?_
      /-
        case mk.mk.mk.intro.intro.refine_1
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁✝ X₂✝ X₃✝ : CochainComplex C Int
        f : Quiver.Hom X₁✝ X₂✝
        g : Quiver.Hom X₂✝ X₃✝
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X₁ X₂ X₃ : CochainComplex C Int
        u₁₂ : Quiver.Hom X₁ X₂
        u₂₃ : Quiver.Hom X₂ X₃
        α : Quiver.Hom (CochainComplex.mappingCone.triangle u₁₂) (CochainComplex.mappi …
        β : Quiver.Hom (CochainComplex.mappingCone.triangle (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient C (Comple …
      -/
    · exact ((quotient _ _).mapTriangle.map α).comm₂
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.intro.intro.refine_2
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁✝ X₂✝ X₃✝ : CochainComplex C Int
        f : Quiver.Hom X₁✝ X₂✝
        g : Quiver.Hom X₂✝ X₃✝
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X₁ X₂ X₃ : CochainComplex C Int
        u₁₂ : Quiver.Hom X₁ X₂
        u₂₃ : Quiver.Hom X₂ X₃
        α : Quiver.Hom (CochainComplex.mappingCone.triangle u₁₂) (CochainComplex.mappi …
        β : Quiver.Hom (CochainComplex.mappingCone.triangle (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient C (Comple …
      -/
    · exact ((quotient _ _).mapTriangle.map α).comm₃.symm.trans (by dsimp [α]; simp)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.intro.intro.refine_3
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁✝ X₂✝ X₃✝ : CochainComplex C Int
        f : Quiver.Hom X₁✝ X₂✝
        g : Quiver.Hom X₂✝ X₃✝
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X₁ X₂ X₃ : CochainComplex C Int
        u₁₂ : Quiver.Hom X₁ X₂
        u₂₃ : Quiver.Hom X₂ X₃
        α : Quiver.Hom (CochainComplex.mappingCone.triangle u₁₂) (CochainComplex.mappi …
        β : Quiver.Hom (CochainComplex.mappingCone.triangle (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.quotient C (Comple …
      -/
    · exact ((quotient _ _).mapTriangle.map β).comm₂.trans (by dsimp [β]; simp)
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.intro.intro.refine_4
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁✝ X₂✝ X₃✝ : CochainComplex C Int
        f : Quiver.Hom X₁✝ X₂✝
        g : Quiver.Hom X₂✝ X₃✝
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X₁ X₂ X₃ : CochainComplex C Int
        u₁₂ : Quiver.Hom X₁ X₂
        u₂₃ : Quiver.Hom X₂ X₃
        α : Quiver.Hom (CochainComplex.mappingCone.triangle u₁₂) (CochainComplex.mappi …
        β : Quiver.Hom (CochainComplex.mappingCone.triangle (CategoryTheory.CategorySt …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · exact ((quotient _ _).mapTriangle.map β).comm₃
      /-
        🎉 no goals
      -/
      /-
        case mk.mk.mk.intro.intro.refine_5
        C : Type u_1
        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : CategoryTheory.Limits.HasBinaryBiproducts C
        X₁✝ X₂✝ X₃✝ : CochainComplex C Int
        f : Quiver.Hom X₁✝ X₂✝
        g : Quiver.Hom X₂✝ X₃✝
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X₁ X₂ X₃ : CochainComplex C Int
        u₁₂ : Quiver.Hom X₁ X₂
        u₂₃ : Quiver.Hom X₂ X₃
        α : Quiver.Hom (CochainComplex.mappingCone.triangle u₁₂) (CochainComplex.mappi …
        β : Quiver.Hom (CochainComplex.mappingCone.triangle (CategoryTheory.CategorySt …
        ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
      -/
    · refine isomorphic_distinguished _ (mappingConeCompTriangleh_distinguished u₁₂ u₂₃) _ ?_
      exact Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _)
        (by dsimp [α, mappingConeCompTriangleh]; simp)
        (by dsimp [β, mappingConeCompTriangleh]; simp)
        (by dsimp [mappingConeCompTriangleh]; simp))


