/-- The limit cone for the product with a zero object. -/
def binaryFanZeroLeft (X : C) : BinaryFan (0 : C) X :=
  BinaryFan.mk 0 (𝟙 X)


/-- The limit cone for the product with a zero object is limiting. -/
def binaryFanZeroLeftIsLimit (X : C) : IsLimit (binaryFanZeroLeft X) :=
                                                     /-
                                                       C : Type u_1
                                                       inst✝² : CategoryTheory.Category.{?u.487, u_1} C
                                                       inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                       X : C
                                                       ⊢ ∀ (s : CategoryTheory.Limits.BinaryFan 0 X), Eq (CategoryTheory.CategoryStru …
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  BinaryFan.isLimitMk (fun s => BinaryFan.snd s) (by aesop_cat) (by aesop_cat)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                        /-
                          C : Type u_1
                          inst✝² : CategoryTheory.Category.{?u.487, u_1} C
                          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          X : C
                          s : CategoryTheory.Limits.BinaryFan 0 X
                          m : Quiver.Hom s.pt X
                          x✝ : Eq (CategoryTheory.CategoryStruct.comp m 0) s.fst
                          h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.i …
                          ⊢ Eq m ((fun s => s.snd) s)
                        -/
    (fun s m _ h₂ => by simpa using h₂)
                        /-
                          🎉 no goals
                        -/


instance hasBinaryProduct_zero_left (X : C) : HasBinaryProduct (0 : C) X :=
  HasLimit.mk ⟨_, binaryFanZeroLeftIsLimit X⟩


/-- A zero object is a left unit for categorical product. -/
def zeroProdIso (X : C) : (0 : C) ⨯ X ≅ X :=
  limit.isoLimitCone ⟨_, binaryFanZeroLeftIsLimit X⟩


@[simp]
theorem zeroProdIso_hom (X : C) : (zeroProdIso X).hom = prod.snd :=
  rfl


@[simp]
theorem zeroProdIso_inv_snd (X : C) : (zeroProdIso X).inv ≫ prod.snd = 𝟙 X := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.zeroProdIso X) …
  -/
  dsimp [zeroProdIso, binaryFanZeroLeft]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The limit cone for the product with a zero object. -/
def binaryFanZeroRight (X : C) : BinaryFan X (0 : C) :=
  BinaryFan.mk (𝟙 X) 0


/-- The limit cone for the product with a zero object is limiting. -/
def binaryFanZeroRightIsLimit (X : C) : IsLimit (binaryFanZeroRight X) :=
                                                     /-
                                                       C : Type u_1
                                                       inst✝² : CategoryTheory.Category.{?u.4805, u_1} C
                                                       inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                       X : C
                                                       ⊢ ∀ (s : CategoryTheory.Limits.BinaryFan X 0), Eq (CategoryTheory.CategoryStru …
                                                     -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  BinaryFan.isLimitMk (fun s => BinaryFan.fst s) (by aesop_cat) (by aesop_cat)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                        /-
                          C : Type u_1
                          inst✝² : CategoryTheory.Category.{?u.4805, u_1} C
                          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          X : C
                          s : CategoryTheory.Limits.BinaryFan X 0
                          m : Quiver.Hom s.pt X
                          h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.i …
                          x✝ : Eq (CategoryTheory.CategoryStruct.comp m 0) s.snd
                          ⊢ Eq m ((fun s => s.fst) s)
                        -/
    (fun s m h₁ _ => by simpa using h₁)
                        /-
                          🎉 no goals
                        -/


instance hasBinaryProduct_zero_right (X : C) : HasBinaryProduct X (0 : C) :=
  HasLimit.mk ⟨_, binaryFanZeroRightIsLimit X⟩


/-- A zero object is a right unit for categorical product. -/
def prodZeroIso (X : C) : X ⨯ (0 : C) ≅ X :=
  limit.isoLimitCone ⟨_, binaryFanZeroRightIsLimit X⟩


@[simp]
theorem prodZeroIso_hom (X : C) : (prodZeroIso X).hom = prod.fst :=
  rfl


@[simp]
theorem prodZeroIso_iso_inv_snd (X : C) : (prodZeroIso X).inv ≫ prod.fst = 𝟙 X := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prodZeroIso X) …
  -/
  dsimp [prodZeroIso, binaryFanZeroRight]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The colimit cocone for the coproduct with a zero object. -/
def binaryCofanZeroLeft (X : C) : BinaryCofan (0 : C) X :=
  BinaryCofan.mk 0 (𝟙 X)


/-- The colimit cocone for the coproduct with a zero object is colimiting. -/
def binaryCofanZeroLeftIsColimit (X : C) : IsColimit (binaryCofanZeroLeft X) :=
                                                           /-
                                                             C : Type u_1
                                                             inst✝² : CategoryTheory.Category.{?u.9135, u_1} C
                                                             inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             X : C
                                                             ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan 0 X), Eq (CategoryTheory.CategorySt …
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  BinaryCofan.isColimitMk (fun s => BinaryCofan.inr s) (by aesop_cat) (by aesop_cat)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                        /-
                          C : Type u_1
                          inst✝² : CategoryTheory.Category.{?u.9135, u_1} C
                          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          X : C
                          s : CategoryTheory.Limits.BinaryCofan 0 X
                          m : Quiver.Hom X s.pt
                          x✝ : Eq (CategoryTheory.CategoryStruct.comp 0 m) s.inl
                          h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
                          ⊢ Eq m ((fun s => s.inr) s)
                        -/
    (fun s m _ h₂ => by simpa using h₂)
                        /-
                          🎉 no goals
                        -/


instance hasBinaryCoproduct_zero_left (X : C) : HasBinaryCoproduct (0 : C) X :=
  HasColimit.mk ⟨_, binaryCofanZeroLeftIsColimit X⟩


/-- A zero object is a left unit for categorical coproduct. -/
def zeroCoprodIso (X : C) : (0 : C) ⨿ X ≅ X :=
  colimit.isoColimitCocone ⟨_, binaryCofanZeroLeftIsColimit X⟩


@[simp]
theorem inr_zeroCoprodIso_hom (X : C) : coprod.inr ≫ (zeroCoprodIso X).hom = 𝟙 X := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
  -/
  dsimp [zeroCoprodIso, binaryCofanZeroLeft]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem zeroCoprodIso_inv (X : C) : (zeroCoprodIso X).inv = coprod.inr :=
  rfl


/-- The colimit cocone for the coproduct with a zero object. -/
def binaryCofanZeroRight (X : C) : BinaryCofan X (0 : C) :=
  BinaryCofan.mk (𝟙 X) 0


/-- The colimit cocone for the coproduct with a zero object is colimiting. -/
def binaryCofanZeroRightIsColimit (X : C) : IsColimit (binaryCofanZeroRight X) :=
                                                           /-
                                                             C : Type u_1
                                                             inst✝² : CategoryTheory.Category.{?u.13877, u_1} C
                                                             inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             X : C
                                                             ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X 0), Eq (CategoryTheory.CategorySt …
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  BinaryCofan.isColimitMk (fun s => BinaryCofan.inl s) (by aesop_cat) (by aesop_cat)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                        /-
                          C : Type u_1
                          inst✝² : CategoryTheory.Category.{?u.13877, u_1} C
                          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          X : C
                          s : CategoryTheory.Limits.BinaryCofan X 0
                          m : Quiver.Hom X s.pt
                          h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
                          x✝ : Eq (CategoryTheory.CategoryStruct.comp 0 m) s.inr
                          ⊢ Eq m ((fun s => s.inl) s)
                        -/
    (fun s m h₁ _ => by simpa using h₁)
                        /-
                          🎉 no goals
                        -/


instance hasBinaryCoproduct_zero_right (X : C) : HasBinaryCoproduct X (0 : C) :=
  HasColimit.mk ⟨_, binaryCofanZeroRightIsColimit X⟩


/-- A zero object is a right unit for categorical coproduct. -/
def coprodZeroIso (X : C) : X ⨿ (0 : C) ≅ X :=
  colimit.isoColimitCocone ⟨_, binaryCofanZeroRightIsColimit X⟩


@[simp]
theorem inr_coprodZeroIso_hom (X : C) : coprod.inl ≫ (coprodZeroIso X).hom = 𝟙 X := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
  -/
  dsimp [coprodZeroIso, binaryCofanZeroRight]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem coprodZeroIso_inv (X : C) : (coprodZeroIso X).inv = coprod.inl :=
  rfl


instance hasPullback_over_zero (X Y : C) [HasBinaryProduct X Y] :
    HasPullback (0 : X ⟶ 0) (0 : Y ⟶ 0) :=
  HasLimit.mk
    ⟨_, isPullbackOfIsTerminalIsProduct _ _ _ _ HasZeroObject.zeroIsTerminal (prodIsProd X Y)⟩


/-- The pullback over the zero object is the product. -/
def pullbackZeroZeroIso (X Y : C) [HasBinaryProduct X Y] :
    pullback (0 : X ⟶ 0) (0 : Y ⟶ 0) ≅ X ⨯ Y :=
  limit.isoLimitCone
    ⟨_, isPullbackOfIsTerminalIsProduct _ _ _ _ HasZeroObject.zeroIsTerminal (prodIsProd X Y)⟩


@[simp]
theorem pullbackZeroZeroIso_inv_fst (X Y : C) [HasBinaryProduct X Y] :
    (pullbackZeroZeroIso X Y).inv ≫ pullback.fst 0 0 = prod.fst := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackZeroZe …
  -/
  dsimp [pullbackZeroZeroIso]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem pullbackZeroZeroIso_inv_snd (X Y : C) [HasBinaryProduct X Y] :
    (pullbackZeroZeroIso X Y).inv ≫ pullback.snd 0 0 = prod.snd := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackZeroZe …
  -/
  dsimp [pullbackZeroZeroIso]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.isoLimit …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem pullbackZeroZeroIso_hom_fst (X Y : C) [HasBinaryProduct X Y] :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
                                                                        inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        X Y : C
                                                                        inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackZeroZe …
                                                                      -/
    (pullbackZeroZeroIso X Y).hom ≫ prod.fst = pullback.fst 0 0 := by simp [← Iso.eq_inv_comp]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem pullbackZeroZeroIso_hom_snd (X Y : C) [HasBinaryProduct X Y] :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
                                                                        inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        X Y : C
                                                                        inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackZeroZe …
                                                                      -/
    (pullbackZeroZeroIso X Y).hom ≫ prod.snd = pullback.snd 0 0 := by simp [← Iso.eq_inv_comp]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance hasPushout_over_zero (X Y : C) [HasBinaryCoproduct X Y] :
    HasPushout (0 : 0 ⟶ X) (0 : 0 ⟶ Y) :=
  HasColimit.mk
    ⟨_, isPushoutOfIsInitialIsCoproduct _ _ _ _ HasZeroObject.zeroIsInitial (coprodIsCoprod X Y)⟩


/-- The pushout over the zero object is the coproduct. -/
def pushoutZeroZeroIso (X Y : C) [HasBinaryCoproduct X Y] :
    pushout (0 : 0 ⟶ X) (0 : 0 ⟶ Y) ≅ X ⨿ Y :=
  colimit.isoColimitCocone
    ⟨_, isPushoutOfIsInitialIsCoproduct _ _ _ _ HasZeroObject.zeroIsInitial (coprodIsCoprod X Y)⟩


@[simp]
theorem inl_pushoutZeroZeroIso_hom (X Y : C) [HasBinaryCoproduct X Y] :
    pushout.inl _ _ ≫ (pushoutZeroZeroIso X Y).hom = coprod.inl := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl 0  …
  -/
  dsimp [pushoutZeroZeroIso]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl 0  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem inr_pushoutZeroZeroIso_hom (X Y : C) [HasBinaryCoproduct X Y] :
    pushout.inr _ _ ≫ (pushoutZeroZeroIso X Y).hom = coprod.inr := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr 0  …
  -/
  dsimp [pushoutZeroZeroIso]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr 0  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem inl_pushoutZeroZeroIso_inv (X Y : C) [HasBinaryCoproduct X Y] :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
                                                                        inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        X Y : C
                                                                        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
                                                                      -/
    coprod.inl ≫ (pushoutZeroZeroIso X Y).inv = pushout.inl _ _ := by simp [Iso.comp_inv_eq]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem inr_pushoutZeroZeroIso_inv (X Y : C) [HasBinaryCoproduct X Y] :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝³ : CategoryTheory.Category.{u_2, u_1} C
                                                                        inst✝² : CategoryTheory.Limits.HasZeroObject C
                                                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        X Y : C
                                                                        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
                                                                      -/
    coprod.inr ≫ (pushoutZeroZeroIso X Y).inv = pushout.inr _ _ := by simp [Iso.comp_inv_eq]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


