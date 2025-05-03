/-- Auxiliary definition for `Functor.ofCocone`. -/
def obj (i : J) : C :=
  if hi : i < j then
    F.obj ⟨i, hi⟩
  else c.pt


/-- Auxiliary definition for `Functor.ofCocone`. -/
def objIso (i : J) (hi : i < j) :
    obj c i ≅ F.obj ⟨i, hi⟩ :=
  eqToIso (dif_pos hi)


/-- Auxiliary definition for `Functor.ofCocone`. -/
def objIsoPt :
    obj c j  ≅ c.pt :=
                       /-
                         C : Type u_1
                         inst✝¹ : CategoryTheory.Category.{?u.4525, u_1} C
                         J : Type u
                         inst✝ : LinearOrder J
                         j : J
                         F : CategoryTheory.Functor (↑(Set.Iio j)) C
                         c : CategoryTheory.Limits.Cocone F
                         ⊢ Not (LT.lt j j)
                       -/
  eqToIso (dif_neg (by simp))
                       /-
                         🎉 no goals
                       -/


/-- Auxiliary definition for `Functor.ofCocone`. -/
def map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    obj c i₁ ⟶ obj c i₂ :=
  if h₂ : i₂ < j then
    (objIso c i₁ (lt_of_le_of_lt hi h₂)).hom ≫ F.map (homOfLE hi) ≫ (objIso c i₂ h₂).inv
  else
                                             /-
                                               C : Type u_1
                                               inst✝¹ : CategoryTheory.Category.{?u.6253, u_1} C
                                               J : Type u
                                               inst✝ : LinearOrder J
                                               j : J
                                               F : CategoryTheory.Functor (↑(Set.Iio j)) C
                                               c : CategoryTheory.Limits.Cocone F
                                               i₁ i₂ : J
                                               hi : LE.le i₁ i₂
                                               hi₂ : LE.le i₂ j
                                               h₂ : Not (LT.lt i₂ j)
                                               ⊢ LE.le j i₂
                                             -/
    have h₂' : i₂ = j := le_antisymm hi₂ (by simpa using h₂)
                                             /-
                                               🎉 no goals
                                             -/
    if h₁ : i₁ < j then
                                                                               /-
                                                                                 C : Type u_1
                                                                                 inst✝¹ : CategoryTheory.Category.{?u.6253, u_1} C
                                                                                 J : Type u
                                                                                 inst✝ : LinearOrder J
                                                                                 j : J
                                                                                 F : CategoryTheory.Functor (↑(Set.Iio j)) C
                                                                                 c : CategoryTheory.Limits.Cocone F
                                                                                 i₁ i₂ : J
                                                                                 hi : LE.le i₁ i₂
                                                                                 hi₂ : LE.le i₂ j
                                                                                 h₂ : Not (LT.lt i₂ j)
                                                                                 h₂' : Eq i₂ j
                                                                                 h₁ : LT.lt i₁ j
                                                                                 ⊢ Eq (CategoryTheory.Functor.ofCocone.obj c j) (CategoryTheory.Functor.ofCocon …
                                                                               -/
      (objIso c i₁ h₁).hom ≫ c.ι.app ⟨i₁, h₁⟩ ≫ (objIsoPt c).inv ≫ eqToHom (by subst h₂'; rfl)
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/
    else
                                                          /-
                                                            C : Type u_1
                                                            inst✝¹ : CategoryTheory.Category.{?u.6253, u_1} C
                                                            J : Type u
                                                            inst✝ : LinearOrder J
                                                            j : J
                                                            F : CategoryTheory.Functor (↑(Set.Iio j)) C
                                                            c : CategoryTheory.Limits.Cocone F
                                                            i₁ i₂ : J
                                                            hi : LE.le i₁ i₂
                                                            hi₂ : LE.le i₂ j
                                                            h₂ : Not (LT.lt i₂ j)
                                                            h₂' : Eq i₂ j
                                                            h₁ : Not (LT.lt i₁ j)
                                                            ⊢ LE.le j i₁
                                                          -/
      have h₁' : i₁ = j := le_antisymm (hi.trans hi₂) (by simpa using h₁)
                                                          /-
                                                            🎉 no goals
                                                          -/
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.6253, u_1} C
                    J : Type u
                    inst✝ : LinearOrder J
                    j : J
                    F : CategoryTheory.Functor (↑(Set.Iio j)) C
                    c : CategoryTheory.Limits.Cocone F
                    i₁ i₂ : J
                    hi : LE.le i₁ i₂
                    hi₂ : LE.le i₂ j
                    h₂ : Not (LT.lt i₂ j)
                    h₂' : Eq i₂ j
                    h₁ : Not (LT.lt i₁ j)
                    h₁' : Eq i₁ j
                    ⊢ Eq (CategoryTheory.Functor.ofCocone.obj c i₁) (CategoryTheory.Functor.ofCoco …
                  -/
      eqToHom (by subst h₁' h₂'; rfl)
                                 /-
                                   🎉 no goals
                                 -/


lemma map_id (i : J) (hi : i ≤ j) :
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.10818, u_1} C
                    J : Type u
                    inst✝ : LinearOrder J
                    j : J
                    F : CategoryTheory.Functor (↑(Set.Iio j)) C
                    c : CategoryTheory.Limits.Cocone F
                    i : J
                    hi : LE.le i j
                    ⊢ LE.le i i
                  -/
    map c i i (by rfl) hi = 𝟙 _:= by
                  /-
                    🎉 no goals
                  -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i : J
    hi : LE.le i j
    ⊢ Eq (CategoryTheory.Functor.ofCocone.map c i i ⋯ hi) (CategoryTheory.Category …
  -/
  dsimp [map]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i : J
    hi : LE.le i j
    ⊢ Eq (dite (LT.lt i j) (fun h₂ => CategoryTheory.CategoryStruct.comp (Category …
  -/
  obtain hi' | rfl := hi.lt_or_eq
    /-
      case inl
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝ : LinearOrder J
      j : J
      F : CategoryTheory.Functor (↑(Set.Iio j)) C
      c : CategoryTheory.Limits.Cocone F
      i : J
      hi : LE.le i j
      hi' : LT.lt i j
      ⊢ Eq (dite (LT.lt i j) (fun h₂ => CategoryTheory.CategoryStruct.comp (Category …
    -/
  · rw [dif_pos hi', F.map_id, id_comp, Iso.hom_inv_id]
    /-
      🎉 no goals
    -/
    /-
      case inr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝ : LinearOrder J
      i : J
      F : CategoryTheory.Functor (↑(Set.Iio i)) C
      c : CategoryTheory.Limits.Cocone F
      hi : LE.le i i
      ⊢ Eq (dite (LT.lt i i) (fun h₂ => CategoryTheory.CategoryStruct.comp (Category …
    -/
  · rw [dif_neg (by simp), dif_neg (by simp)]
    /-
      🎉 no goals
    -/


lemma map_comp (i₁ i₂ i₃ : J) (hi : i₁ ≤ i₂) (hi' : i₂ ≤ i₃) (hi₃ : i₃ ≤ j) :
    map c i₁ i₃ (hi.trans hi') hi₃ =
      map c i₁ i₂ hi (hi'.trans hi₃) ≫
        map c i₂ i₃ hi' hi₃ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i₁ i₂ i₃ : J
    hi : LE.le i₁ i₂
    hi' : LE.le i₂ i₃
    hi₃ : LE.le i₃ j
    ⊢ Eq (CategoryTheory.Functor.ofCocone.map c i₁ i₃ ⋯ hi₃) (CategoryTheory.Categ …
  -/
  obtain hi₁₂ | rfl := hi.lt_or_eq
    /-
      case inl
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝ : LinearOrder J
      j : J
      F : CategoryTheory.Functor (↑(Set.Iio j)) C
      c : CategoryTheory.Limits.Cocone F
      i₁ i₂ i₃ : J
      hi : LE.le i₁ i₂
      hi' : LE.le i₂ i₃
      hi₃ : LE.le i₃ j
      hi₁₂ : LT.lt i₁ i₂
      ⊢ Eq (CategoryTheory.Functor.ofCocone.map c i₁ i₃ ⋯ hi₃) (CategoryTheory.Categ …
    -/
  · obtain hi₂₃ | rfl := hi'.lt_or_eq
      /-
        case inl.inl
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝ : LinearOrder J
        j : J
        F : CategoryTheory.Functor (↑(Set.Iio j)) C
        c : CategoryTheory.Limits.Cocone F
        i₁ i₂ i₃ : J
        hi : LE.le i₁ i₂
        hi' : LE.le i₂ i₃
        hi₃ : LE.le i₃ j
        hi₁₂ : LT.lt i₁ i₂
        hi₂₃ : LT.lt i₂ i₃
        ⊢ Eq (CategoryTheory.Functor.ofCocone.map c i₁ i₃ ⋯ hi₃) (CategoryTheory.Categ …
      -/
    · dsimp [map]
      /-
        case inl.inl
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝ : LinearOrder J
        j : J
        F : CategoryTheory.Functor (↑(Set.Iio j)) C
        c : CategoryTheory.Limits.Cocone F
        i₁ i₂ i₃ : J
        hi : LE.le i₁ i₂
        hi' : LE.le i₂ i₃
        hi₃ : LE.le i₃ j
        hi₁₂ : LT.lt i₁ i₂
        hi₂₃ : LT.lt i₂ i₃
        ⊢ Eq (dite (LT.lt i₃ j) (fun h₂ => CategoryTheory.CategoryStruct.comp (Categor …
      -/
      obtain hi₃' | rfl := hi₃.lt_or_eq
      · rw [dif_pos hi₃', dif_pos (hi₂₃.trans hi₃'), dif_pos hi₃', assoc, assoc,
          Iso.inv_hom_id_assoc, ← Functor.map_comp_assoc, homOfLE_comp]
      · rw [dif_neg (by simp), dif_pos (hi₁₂.trans hi₂₃), dif_pos hi₂₃, dif_neg (by simp),
          dif_pos hi₂₃, eqToHom_refl, comp_id, assoc, assoc, Iso.inv_hom_id_assoc,
          Cocone.w_assoc]
      /-
        case inl.inr
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝ : LinearOrder J
        j : J
        F : CategoryTheory.Functor (↑(Set.Iio j)) C
        c : CategoryTheory.Limits.Cocone F
        i₁ i₂ : J
        hi : LE.le i₁ i₂
        hi₁₂ : LT.lt i₁ i₂
        hi' : LE.le i₂ i₂
        hi₃ : LE.le i₂ j
        ⊢ Eq (CategoryTheory.Functor.ofCocone.map c i₁ i₂ ⋯ hi₃) (CategoryTheory.Categ …
      -/
    · rw [map_id, comp_id]
      /-
        🎉 no goals
      -/
    /-
      case inr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝ : LinearOrder J
      j : J
      F : CategoryTheory.Functor (↑(Set.Iio j)) C
      c : CategoryTheory.Limits.Cocone F
      i₁ i₃ : J
      hi₃ : LE.le i₃ j
      hi : LE.le i₁ i₁
      hi' : LE.le i₁ i₃
      ⊢ Eq (CategoryTheory.Functor.ofCocone.map c i₁ i₃ ⋯ hi₃) (CategoryTheory.Categ …
    -/
  · rw [map_id, id_comp]
    /-
      🎉 no goals
    -/


/-- Given a functor `F : Set.Iio j ⥤ C` and a cocone `c : Cocone F`,
where `j : J` and `J` is linearly ordered, this is the functor
`Set.Iic j ⥤ C` which extends `F` and sends the top element to `c.pt`. -/
def ofCocone : Set.Iic j ⥤ C where
  obj i := ofCocone.obj c i.1
  map {_ j} f := ofCocone.map c _ _ (leOfHom f) j.2
  map_id i := ofCocone.map_id _ _ i.2
  map_comp {_ _ i₃} _ _ := ofCocone.map_comp _ _ _ _ _ _ i₃.2


/-- The isomorphism `(ofCocone c).obj ⟨i, _⟩ ≅ F.obj ⟨i, _⟩` when `i < j`. -/
def ofCoconeObjIso (i : J) (hi : i < j) :
    (ofCocone c).obj ⟨i, hi.le⟩ ≅ F.obj ⟨i, hi⟩ :=
  ofCocone.objIso c _ _


/-- The isomorphism `(ofCocone c).obj ⟨j, _⟩ ≅ c.pt`. -/
def ofCoconeObjIsoPt :
                            /-
                              C : Type u_1
                              inst✝¹ : CategoryTheory.Category.{?u.28566, u_1} C
                              J : Type u
                              inst✝ : LinearOrder J
                              j : J
                              F : CategoryTheory.Functor (↑(Set.Iio j)) C
                              c : CategoryTheory.Limits.Cocone F
                              ⊢ Membership.mem (Set.Iic j) j
                            -/
    (ofCocone c).obj ⟨j, by simp⟩ ≅ c.pt :=
                            /-
                              🎉 no goals
                            -/
  ofCocone.objIsoPt c


lemma ofCocone_map_to_top (i : J) (hi : i < j) :
    (ofCocone c).map (homOfLE hi.le) =
      (ofCoconeObjIso c i hi).hom ≫ c.ι.app ⟨i, hi⟩ ≫ (ofCoconeObjIsoPt c).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i : J
    hi : LT.lt i j
    ⊢ Eq ((CategoryTheory.Functor.ofCocone c).map (CategoryTheory.homOfLE ⋯)) (Cat …
  -/
  dsimp [ofCocone, ofCocone.map, ofCoconeObjIso, ofCoconeObjIsoPt]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i : J
    hi : LT.lt i j
    ⊢ Eq (dite (LT.lt j j) (fun h₂ => CategoryTheory.CategoryStruct.comp (Category …
  -/
  rw [dif_neg (by simp), dif_pos hi, comp_id]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma ofCocone_map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ < j) :
    (ofCocone c).map (homOfLE hi : ⟨i₁, hi.trans hi₂.le⟩ ⟶ ⟨i₂, hi₂.le⟩) =
      (ofCoconeObjIso c i₁ (lt_of_le_of_lt hi hi₂)).hom ≫ F.map (homOfLE hi) ≫
        (ofCoconeObjIso c i₂ hi₂).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i₁ i₂ : J
    hi : LE.le i₁ i₂
    hi₂ : LT.lt i₂ j
    ⊢ Eq ((CategoryTheory.Functor.ofCocone c).map (CategoryTheory.homOfLE hi)) (Ca …
  -/
  dsimp [ofCocone, ofCoconeObjIso, ofCocone.map]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i₁ i₂ : J
    hi : LE.le i₁ i₂
    hi₂ : LT.lt i₂ j
    ⊢ Eq (dite (LT.lt i₂ j) (fun h₂ => CategoryTheory.CategoryStruct.comp (Categor …
  -/
  rw [dif_pos hi₂]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma ofCoconeObjIso_hom_naturality (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ < j) :
    (ofCocone c).map (homOfLE hi : ⟨i₁, hi.trans hi₂.le⟩ ⟶ ⟨i₂, hi₂.le⟩) ≫
      (ofCoconeObjIso c i₂ hi₂).hom =
      (ofCoconeObjIso c i₁ (lt_of_le_of_lt hi hi₂)).hom ≫ F.map (homOfLE hi) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝ : LinearOrder J
    j : J
    F : CategoryTheory.Functor (↑(Set.Iio j)) C
    c : CategoryTheory.Limits.Cocone F
    i₁ i₂ : J
    hi : LE.le i₁ i₂
    hi₂ : LT.lt i₂ j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.ofCocone c). …
  -/
  rw [ofCocone_map c i₁ i₂ hi hi₂, assoc, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


/-- The isomorphism expressing that `ofCocone c` extends the functor `F`
when `c : Cocone F`. -/
@[simps!]
def restrictionLTOfCoconeIso :
    Iteration.restrictionLT (ofCocone c) (Preorder.le_refl j) ≅ F :=
  NatIso.ofComponents (fun ⟨i, hi⟩ ↦ ofCoconeObjIso c i hi)
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.43750, u_1} C
          J : Type u
          inst✝ : LinearOrder J
          j : J
          F : CategoryTheory.Functor (↑(Set.Iio j)) C
          c : CategoryTheory.Limits.Cocone F
          ⊢ ∀ {X Y : ↑(Set.Iio j)} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStru …
        -/
    (by intros; apply ofCoconeObjIso_hom_naturality)
                /-
                  🎉 no goals
                -/


variable {c} in
/-- If `c` is a colimit cocone, then so is `coconeOfLE (ofCocone c) (Preorder.le_refl j)`. -/
def isColimitCoconeOfLEOfCocone (hc : IsColimit c) :
    IsColimit (Iteration.coconeOfLE (ofCocone c) (Preorder.le_refl j)) :=
  (IsColimit.precomposeInvEquiv (restrictionLTOfCoconeIso c) _).1
    (IsColimit.ofIsoColimit hc
      (Cocones.ext (ofCoconeObjIsoPt c).symm (fun ⟨i, hi⟩ ↦ by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.47362, u_1} C
          J : Type u
          inst✝ : LinearOrder J
          j : J
          F : CategoryTheory.Functor (↑(Set.Iio j)) C
          c : CategoryTheory.Limits.Cocone F
          hc : CategoryTheory.Limits.IsColimit c
          x✝ : ↑(Set.Iio j)
          i : J
          hi : Membership.mem (Set.Iio j) i
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app ⟨i, hi⟩) (CategoryTheory.Fun …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.47362, u_1} C
          J : Type u
          inst✝ : LinearOrder J
          j : J
          F : CategoryTheory.Functor (↑(Set.Iio j)) C
          c : CategoryTheory.Limits.Cocone F
          hc : CategoryTheory.Limits.IsColimit c
          x✝ : ↑(Set.Iio j)
          i : J
          hi : Membership.mem (Set.Iio j) i
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app ⟨i, hi⟩) (CategoryTheory.Fun …
        -/
        rw [ofCocone_map_to_top _ _ hi, Iso.inv_hom_id_assoc])))
        /-
          🎉 no goals
        -/


