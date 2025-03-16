pe u_1
                                                  inst✝² : CategoryTheory.Category.{?u.3310, u_1} C
                                                  J : Type u
                                                  inst✝¹ : LinearOrder J
                                                  inst✝ : SuccOrder J
                                                  j : J
                                                  hj : Not (IsMax j)
                                                  F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                  X : C
                                                  ⊢ Membership.mem (Set.Iic j) j
                                                -/
  (F : Set.Iic j ⥤ C) {X : C} (τ : F.obj ⟨j, by simp⟩ ⟶ X)
                                                /-
                                                  🎉 no goals
                                                -/

namespace extendToSucc

variable (X)

/-- `extendToSucc`, on objects: it coincides with `F.obj` for `i ≤ j`, and
it sends `Order.succ j` to the given object `X`. -/
def obj (i : Set.Iic (Order.succ j)) : C :=
  if hij : i.1 ≤ j then F.obj ⟨i.1, hij⟩ else X


¹ : LinearOrder J
                                                  inst✝ : SuccOrder J
                                                  j : J
                                                  hj : Not (IsMax j)
                                                  F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                  X : C
                                                  ⊢ Membership.mem (Set.Iic j) j
                                                -/
  (F : Set.Iic j ⥤ C) {X : C} (τ : F.obj ⟨j, by simp⟩ ⟶ X)
                                                /-
                                                  🎉 no goals
                                                -/

namespace extendToSucc

variable (X)

/-- `extendToSucc`, on objects: it coincides with `F.obj` for `i ≤ j`, and
it sends `Order.succ j` to the given object `X`. -/
def obj (i : Set.Iic (Order.succ j)) : C :=
  if hij : i.1 ≤ j then F.obj ⟨i.1, hij⟩ else X

/-- The isomorphism `obj F X ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j`. -/
def objIso (i : Set.Iic j) :
    obj F X ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i := eqToIso (dif_pos i.2)


   hj : Not (IsMax j)
                                                  F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                  X : C
                                                  ⊢ Membership.mem (Set.Iic j) j
                                                -/
  (F : Set.Iic j ⥤ C) {X : C} (τ : F.obj ⟨j, by simp⟩ ⟶ X)
                                                /-
                                                  🎉 no goals
                                                -/

namespace extendToSucc

variable (X)

/-- `extendToSucc`, on objects: it coincides with `F.obj` for `i ≤ j`, and
it sends `Order.succ j` to the given object `X`. -/
def obj (i : Set.Iic (Order.succ j)) : C :=
  if hij : i.1 ≤ j then F.obj ⟨i.1, hij⟩ else X

/-- The isomorphism `obj F X ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j`. -/
def objIso (i : Set.Iic j) :
    obj F X ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i := eqToIso (dif_pos i.2)

/-- The isomorphism `obj F X ⟨Order.succ j, _⟩ ≅ X`. -/
def objSuccIso :
                              /-
                                C : Type u_1
                                inst✝² : CategoryTheory.Category.{?u.8780, u_1} C
                                J : Type u
                                inst✝¹ : LinearOrder J
                                inst✝ : SuccOrder J
                                j : J
                                hj : Not (IsMax j)
                                F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                X : C
                                τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                                ⊢ Membership.mem (Set.Iic (Order.succ j)) (Order.succ j)
                              -/
    obj F X ⟨Order.succ j, by simp⟩ ≅ X :=
                              /-
                                🎉 no goals
                              -/
                       /-
                         C : Type u_1
                         inst✝² : CategoryTheory.Category.{?u.8780, u_1} C
                         J : Type u
                         inst✝¹ : LinearOrder J
                         inst✝ : SuccOrder J
                         j : J
                         hj : Not (IsMax j)
                         F : CategoryTheory.Functor (↑(Set.Iic j)) C
                         X : C
                         τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                         ⊢ Not (LE.le (↑⟨Order.succ j, ⋯⟩) j)
                       -/
  eqToIso (dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj))
                       /-
                         🎉 no goals
                       -/


           ⊢ Membership.mem (Set.Iic j) j
                                                -/
  (F : Set.Iic j ⥤ C) {X : C} (τ : F.obj ⟨j, by simp⟩ ⟶ X)
                                                /-
                                                  🎉 no goals
                                                -/

namespace extendToSucc

variable (X)

/-- `extendToSucc`, on objects: it coincides with `F.obj` for `i ≤ j`, and
it sends `Order.succ j` to the given object `X`. -/
def obj (i : Set.Iic (Order.succ j)) : C :=
  if hij : i.1 ≤ j then F.obj ⟨i.1, hij⟩ else X

/-- The isomorphism `obj F X ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j`. -/
def objIso (i : Set.Iic j) :
    obj F X ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i := eqToIso (dif_pos i.2)

/-- The isomorphism `obj F X ⟨Order.succ j, _⟩ ≅ X`. -/
def objSuccIso :
    obj F X ⟨Order.succ j, by simp⟩ ≅ X :=
  eqToIso (dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj))

variable {X}

/-- `extendToSucc`, on morphisms. -/
def map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ Order.succ j) :
    obj F X ⟨i₁, hi.trans hi₂⟩ ⟶ obj F X ⟨i₂, hi₂⟩ :=
  if h₁ : i₂ ≤ j then
    (objIso F X ⟨i₁, hi.trans h₁⟩).hom ≫ F.map (homOfLE hi) ≫ (objIso F X ⟨i₂, h₁⟩).inv
  else
    if h₂ : i₁ ≤ j then
      (objIso F X ⟨i₁, h₂⟩).hom ≫ F.map (homOfLE h₂) ≫ τ ≫
        (objSuccIso hj F X).inv ≫ eqToHom (by
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.13066, u_1} C
            J : Type u
            inst✝¹ : LinearOrder J
            inst✝ : SuccOrder J
            j : J
            hj : Not (IsMax j)
            F : CategoryTheory.Functor (↑(Set.Iic j)) C
            X : C
            τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
            i₁ i₂ : J
            hi : LE.le i₁ i₂
            hi₂ : LE.le i₂ (Order.succ j)
            h₁ : Not (LE.le i₂ j)
            h₂ : LE.le i₁ j
            ⊢ Eq (CategoryTheory.Functor.extendToSucc.obj F X ⟨Order.succ j, ⋯⟩) (Category …
          -/
          congr
          /-
            case e_i.e_val
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.13066, u_1} C
            J : Type u
            inst✝¹ : LinearOrder J
            inst✝ : SuccOrder J
            j : J
            hj : Not (IsMax j)
            F : CategoryTheory.Functor (↑(Set.Iic j)) C
            X : C
            τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
            i₁ i₂ : J
            hi : LE.le i₁ i₂
            hi₂ : LE.le i₂ (Order.succ j)
            h₁ : Not (LE.le i₂ j)
            h₂ : LE.le i₁ j
            ⊢ Eq (Order.succ j) i₂
          -/
          exact le_antisymm (Order.succ_le_of_lt (not_le.1 h₁)) hi₂)
          /-
            🎉 no goals
          -/
    else
      eqToHom (by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.13066, u_1} C
          J : Type u
          inst✝¹ : LinearOrder J
          inst✝ : SuccOrder J
          j : J
          hj : Not (IsMax j)
          F : CategoryTheory.Functor (↑(Set.Iic j)) C
          X : C
          τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
          i₁ i₂ : J
          hi : LE.le i₁ i₂
          hi₂ : LE.le i₂ (Order.succ j)
          h₁ : Not (LE.le i₂ j)
          h₂ : Not (LE.le i₁ j)
          ⊢ Eq (CategoryTheory.Functor.extendToSucc.obj F X ⟨i₁, ⋯⟩) (CategoryTheory.Fun …
        -/
        congr
        rw [le_antisymm hi₂ (Order.succ_le_of_lt (not_le.1 h₁)),
          le_antisymm (hi.trans hi₂) (Order.succ_le_of_lt (not_le.1 h₂))])


:= eqToIso (dif_pos i.2)

/-- The isomorphism `obj F X ⟨Order.succ j, _⟩ ≅ X`. -/
def objSuccIso :
    obj F X ⟨Order.succ j, by simp⟩ ≅ X :=
  eqToIso (dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj))

variable {X}

/-- `extendToSucc`, on morphisms. -/
def map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ Order.succ j) :
    obj F X ⟨i₁, hi.trans hi₂⟩ ⟶ obj F X ⟨i₂, hi₂⟩ :=
  if h₁ : i₂ ≤ j then
    (objIso F X ⟨i₁, hi.trans h₁⟩).hom ≫ F.map (homOfLE hi) ≫ (objIso F X ⟨i₂, h₁⟩).inv
  else
    if h₂ : i₁ ≤ j then
      (objIso F X ⟨i₁, h₂⟩).hom ≫ F.map (homOfLE h₂) ≫ τ ≫
        (objSuccIso hj F X).inv ≫ eqToHom (by
          congr
          exact le_antisymm (Order.succ_le_of_lt (not_le.1 h₁)) hi₂)
    else
      eqToHom (by
        congr
        rw [le_antisymm hi₂ (Order.succ_le_of_lt (not_le.1 h₁)),
          le_antisymm (hi.trans hi₂) (Order.succ_le_of_lt (not_le.1 h₂))])

lemma map_eq (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    map hj F τ i₁ i₂ hi (hi₂.trans (Order.le_succ j)) =
      (objIso F X ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) ≫
        (objIso F X ⟨i₂, hi₂⟩).inv :=
  dif_pos hi₂



def map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ Order.succ j) :
    obj F X ⟨i₁, hi.trans hi₂⟩ ⟶ obj F X ⟨i₂, hi₂⟩ :=
  if h₁ : i₂ ≤ j then
    (objIso F X ⟨i₁, hi.trans h₁⟩).hom ≫ F.map (homOfLE hi) ≫ (objIso F X ⟨i₂, h₁⟩).inv
  else
    if h₂ : i₁ ≤ j then
      (objIso F X ⟨i₁, h₂⟩).hom ≫ F.map (homOfLE h₂) ≫ τ ≫
        (objSuccIso hj F X).inv ≫ eqToHom (by
          congr
          exact le_antisymm (Order.succ_le_of_lt (not_le.1 h₁)) hi₂)
    else
      eqToHom (by
        congr
        rw [le_antisymm hi₂ (Order.succ_le_of_lt (not_le.1 h₁)),
          le_antisymm (hi.trans hi₂) (Order.succ_le_of_lt (not_le.1 h₂))])

lemma map_eq (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    map hj F τ i₁ i₂ hi (hi₂.trans (Order.le_succ j)) =
      (objIso F X ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) ≫
        (objIso F X ⟨i₂, hi₂⟩).inv :=
  dif_pos hi₂

lemma map_self_succ :
                                                      /-
                                                        C : Type u_1
                                                        inst✝² : CategoryTheory.Category.{?u.21271, u_1} C
                                                        J : Type u
                                                        inst✝¹ : LinearOrder J
                                                        inst✝ : SuccOrder J
                                                        j : J
                                                        hj : Not (IsMax j)
                                                        F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                        X : C
                                                        τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                                                        ⊢ LE.le (Order.succ j) (Order.succ j)
                                                      -/
    map hj F τ j (Order.succ j) (Order.le_succ j) (by rfl) =
                                                      /-
                                                        🎉 no goals
                                                      -/
                         /-
                           C : Type u_1
                           inst✝² : CategoryTheory.Category.{?u.21271, u_1} C
                           J : Type u
                           inst✝¹ : LinearOrder J
                           inst✝ : SuccOrder J
                           j : J
                           hj : Not (IsMax j)
                           F : CategoryTheory.Functor (↑(Set.Iic j)) C
                           X : C
                           τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                           ⊢ Membership.mem (Set.Iic j) j
                         -/
      (objIso F X ⟨j, by simp⟩).hom ≫ τ ≫ (objSuccIso hj F X).inv := by
                         /-
                           🎉 no goals
                         -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ j (Order.succ j) ⋯ ⋯) (Ca …
  -/
  dsimp [map]
  rw [dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj),
    dif_pos (by rfl), map_id, comp_id, id_comp]



    if h₂ : i₁ ≤ j then
      (objIso F X ⟨i₁, h₂⟩).hom ≫ F.map (homOfLE h₂) ≫ τ ≫
        (objSuccIso hj F X).inv ≫ eqToHom (by
          congr
          exact le_antisymm (Order.succ_le_of_lt (not_le.1 h₁)) hi₂)
    else
      eqToHom (by
        congr
        rw [le_antisymm hi₂ (Order.succ_le_of_lt (not_le.1 h₁)),
          le_antisymm (hi.trans hi₂) (Order.succ_le_of_lt (not_le.1 h₂))])

lemma map_eq (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    map hj F τ i₁ i₂ hi (hi₂.trans (Order.le_succ j)) =
      (objIso F X ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) ≫
        (objIso F X ⟨i₂, hi₂⟩).inv :=
  dif_pos hi₂

lemma map_self_succ :
    map hj F τ j (Order.succ j) (Order.le_succ j) (by rfl) =
      (objIso F X ⟨j, by simp⟩).hom ≫ τ ≫ (objSuccIso hj F X).inv := by
  dsimp [map]
  rw [dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj),
    dif_pos (by rfl), map_id, comp_id, id_comp]

@[simp]
lemma map_id (i : J) (hi : i ≤ Order.succ j) :
                       /-
                         C : Type u_1
                         inst✝² : CategoryTheory.Category.{?u.25488, u_1} C
                         J : Type u
                         inst✝¹ : LinearOrder J
                         inst✝ : SuccOrder J
                         j : J
                         hj : Not (IsMax j)
                         F : CategoryTheory.Functor (↑(Set.Iic j)) C
                         X : C
                         τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                         i : J
                         hi : LE.le i (Order.succ j)
                         ⊢ LE.le i i
                       -/
    map hj F τ i i (by rfl) hi = 𝟙 _ := by
                       /-
                         🎉 no goals
                       -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    i : J
    hi : LE.le i (Order.succ j)
    ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i i ⋯ hi) (CategoryTheory …
  -/
  dsimp [map]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    i : J
    hi : LE.le i (Order.succ j)
    ⊢ Eq (dite (LE.le i j) (fun h₁ => CategoryTheory.CategoryStruct.comp (Category …
  -/
  by_cases h₁ : i ≤ j
    /-
      case pos
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝¹ : LinearOrder J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      F : CategoryTheory.Functor (↑(Set.Iic j)) C
      X : C
      τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
      i : J
      hi : LE.le i (Order.succ j)
      h₁ : LE.le i j
      ⊢ Eq (dite (LE.le i j) (fun h₁ => CategoryTheory.CategoryStruct.comp (Category …
    -/
  · rw [dif_pos h₁, CategoryTheory.Functor.map_id, id_comp, Iso.hom_inv_id]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝¹ : LinearOrder J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      F : CategoryTheory.Functor (↑(Set.Iic j)) C
      X : C
      τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
      i : J
      hi : LE.le i (Order.succ j)
      h₁ : Not (LE.le i j)
      ⊢ Eq (dite (LE.le i j) (fun h₁ => CategoryTheory.CategoryStruct.comp (Category …
    -/
  · obtain rfl : i = Order.succ j := le_antisymm hi (Order.succ_le_of_lt (not_le.1 h₁))
    rw [dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj),
      dif_neg h₁]


c_le_of_lt (not_le.1 h₂))])

lemma map_eq (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    map hj F τ i₁ i₂ hi (hi₂.trans (Order.le_succ j)) =
      (objIso F X ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) ≫
        (objIso F X ⟨i₂, hi₂⟩).inv :=
  dif_pos hi₂

lemma map_self_succ :
    map hj F τ j (Order.succ j) (Order.le_succ j) (by rfl) =
      (objIso F X ⟨j, by simp⟩).hom ≫ τ ≫ (objSuccIso hj F X).inv := by
  dsimp [map]
  rw [dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj),
    dif_pos (by rfl), map_id, comp_id, id_comp]

@[simp]
lemma map_id (i : J) (hi : i ≤ Order.succ j) :
    map hj F τ i i (by rfl) hi = 𝟙 _ := by
  dsimp [map]
  by_cases h₁ : i ≤ j
  · rw [dif_pos h₁, CategoryTheory.Functor.map_id, id_comp, Iso.hom_inv_id]
  · obtain rfl : i = Order.succ j := le_antisymm hi (Order.succ_le_of_lt (not_le.1 h₁))
    rw [dif_neg (by simpa only [Order.succ_le_iff_isMax] using hj),
      dif_neg h₁]

lemma map_comp (i₁ i₂ i₃ : J) (h₁₂ : i₁ ≤ i₂) (h₂₃ : i₂ ≤ i₃) (h : i₃ ≤ Order.succ j) :
    map hj F τ i₁ i₃ (h₁₂.trans h₂₃) h =
      map hj F τ i₁ i₂ h₁₂ (h₂₃.trans h) ≫ map hj F τ i₂ i₃ h₂₃ h := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    i₁ i₂ i₃ : J
    h₁₂ : LE.le i₁ i₂
    h₂₃ : LE.le i₂ i₃
    h : LE.le i₃ (Order.succ j)
    ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ i₃ ⋯ h) (CategoryTheor …
  -/
  by_cases h₁ : i₃ ≤ j
  · rw [map_eq hj F τ i₁ i₂ _ (h₂₃.trans h₁), map_eq hj F τ i₂ i₃ _ h₁,
      map_eq hj F τ i₁ i₃ _ h₁, assoc, assoc, Iso.inv_hom_id_assoc, ← map_comp_assoc,
      homOfLE_comp]
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝¹ : LinearOrder J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      F : CategoryTheory.Functor (↑(Set.Iic j)) C
      X : C
      τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
      i₁ i₂ i₃ : J
      h₁₂ : LE.le i₁ i₂
      h₂₃ : LE.le i₂ i₃
      h : LE.le i₃ (Order.succ j)
      h₁ : Not (LE.le i₃ j)
      ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ i₃ ⋯ h) (CategoryTheor …
    -/
  · obtain rfl : i₃ = Order.succ j := le_antisymm h (Order.succ_le_of_lt (not_le.1 h₁))
    /-
      case neg
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      J : Type u
      inst✝¹ : LinearOrder J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      F : CategoryTheory.Functor (↑(Set.Iic j)) C
      X : C
      τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
      i₁ i₂ : J
      h₁₂ : LE.le i₁ i₂
      h₂₃ : LE.le i₂ (Order.succ j)
      h : LE.le (Order.succ j) (Order.succ j)
      h₁ : Not (LE.le (Order.succ j) j)
      ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ (Order.succ j) ⋯ h) (C …
    -/
    obtain h₂ | rfl := h₂₃.lt_or_eq
      /-
        case neg.inl
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝¹ : LinearOrder J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        F : CategoryTheory.Functor (↑(Set.Iic j)) C
        X : C
        τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
        i₁ i₂ : J
        h₁₂ : LE.le i₁ i₂
        h₂₃ : LE.le i₂ (Order.succ j)
        h : LE.le (Order.succ j) (Order.succ j)
        h₁ : Not (LE.le (Order.succ j) j)
        h₂ : LT.lt i₂ (Order.succ j)
        ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ (Order.succ j) ⋯ h) (C …
      -/
    · rw [Order.lt_succ_iff_of_not_isMax hj] at h₂
      /-
        case neg.inl
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝¹ : LinearOrder J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        F : CategoryTheory.Functor (↑(Set.Iic j)) C
        X : C
        τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
        i₁ i₂ : J
        h₁₂ : LE.le i₁ i₂
        h₂₃ : LE.le i₂ (Order.succ j)
        h : LE.le (Order.succ j) (Order.succ j)
        h₁ : Not (LE.le (Order.succ j) j)
        h₂ : LE.le i₂ j
        ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ (Order.succ j) ⋯ h) (C …
      -/
      rw [map_eq hj F τ i₁ i₂ _ h₂]
      /-
        case neg.inl
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝¹ : LinearOrder J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        F : CategoryTheory.Functor (↑(Set.Iic j)) C
        X : C
        τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
        i₁ i₂ : J
        h₁₂ : LE.le i₁ i₂
        h₂₃ : LE.le i₂ (Order.succ j)
        h : LE.le (Order.succ j) (Order.succ j)
        h₁ : Not (LE.le (Order.succ j) j)
        h₂ : LE.le i₂ j
        ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ (Order.succ j) ⋯ h) (C …
      -/
      dsimp [map]
      rw [dif_neg h₁, dif_pos (h₁₂.trans h₂), dif_neg h₁, dif_pos h₂,
        assoc, assoc, Iso.inv_hom_id_assoc,comp_id, ← map_comp_assoc, homOfLE_comp]
      /-
        case neg.inr
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        J : Type u
        inst✝¹ : LinearOrder J
        inst✝ : SuccOrder J
        j : J
        hj : Not (IsMax j)
        F : CategoryTheory.Functor (↑(Set.Iic j)) C
        X : C
        τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
        i₁ : J
        h : LE.le (Order.succ j) (Order.succ j)
        h₁ : Not (LE.le (Order.succ j) j)
        h₁₂ : LE.le i₁ (Order.succ j)
        h₂₃ : LE.le (Order.succ j) (Order.succ j)
        ⊢ Eq (CategoryTheory.Functor.extendToSucc.map hj F τ i₁ (Order.succ j) ⋯ h) (C …
      -/
    · rw [map_id, comp_id]
      /-
        🎉 no goals
      -/


ax] using hj),
      dif_neg h₁]

lemma map_comp (i₁ i₂ i₃ : J) (h₁₂ : i₁ ≤ i₂) (h₂₃ : i₂ ≤ i₃) (h : i₃ ≤ Order.succ j) :
    map hj F τ i₁ i₃ (h₁₂.trans h₂₃) h =
      map hj F τ i₁ i₂ h₁₂ (h₂₃.trans h) ≫ map hj F τ i₂ i₃ h₂₃ h := by
  by_cases h₁ : i₃ ≤ j
  · rw [map_eq hj F τ i₁ i₂ _ (h₂₃.trans h₁), map_eq hj F τ i₂ i₃ _ h₁,
      map_eq hj F τ i₁ i₃ _ h₁, assoc, assoc, Iso.inv_hom_id_assoc, ← map_comp_assoc,
      homOfLE_comp]
  · obtain rfl : i₃ = Order.succ j := le_antisymm h (Order.succ_le_of_lt (not_le.1 h₁))
    obtain h₂ | rfl := h₂₃.lt_or_eq
    · rw [Order.lt_succ_iff_of_not_isMax hj] at h₂
      rw [map_eq hj F τ i₁ i₂ _ h₂]
      dsimp [map]
      rw [dif_neg h₁, dif_pos (h₁₂.trans h₂), dif_neg h₁, dif_pos h₂,
        assoc, assoc, Iso.inv_hom_id_assoc,comp_id, ← map_comp_assoc, homOfLE_comp]
    · rw [map_id, comp_id]

end extendToSucc

open extendToSucc in
include hj in
/-- The extension to `Set.Iic (Order.succ j) ⥤ C` of a functor `F : Set.Iic j ⥤ C`,
when we specify a morphism `F.obj ⟨j, _⟩ ⟶ X`. -/
def extendToSucc : Set.Iic (Order.succ j) ⥤ C where
  obj := obj F X
  map {i₁ i₂} f := map hj F τ i₁ i₂ (leOfHom f) i₂.2
  map_id _ := extendToSucc.map_id _ F τ _ _
  map_comp {i₁ i₂ i₃} f g := extendToSucc.map_comp hj F τ i₁ i₂ i₃ (leOfHom f) (leOfHom g) i₃.2


ssoc, Iso.inv_hom_id_assoc, ← map_comp_assoc,
      homOfLE_comp]
  · obtain rfl : i₃ = Order.succ j := le_antisymm h (Order.succ_le_of_lt (not_le.1 h₁))
    obtain h₂ | rfl := h₂₃.lt_or_eq
    · rw [Order.lt_succ_iff_of_not_isMax hj] at h₂
      rw [map_eq hj F τ i₁ i₂ _ h₂]
      dsimp [map]
      rw [dif_neg h₁, dif_pos (h₁₂.trans h₂), dif_neg h₁, dif_pos h₂,
        assoc, assoc, Iso.inv_hom_id_assoc,comp_id, ← map_comp_assoc, homOfLE_comp]
    · rw [map_id, comp_id]

end extendToSucc

open extendToSucc in
include hj in
/-- The extension to `Set.Iic (Order.succ j) ⥤ C` of a functor `F : Set.Iic j ⥤ C`,
when we specify a morphism `F.obj ⟨j, _⟩ ⟶ X`. -/
def extendToSucc : Set.Iic (Order.succ j) ⥤ C where
  obj := obj F X
  map {i₁ i₂} f := map hj F τ i₁ i₂ (leOfHom f) i₂.2
  map_id _ := extendToSucc.map_id _ F τ _ _
  map_comp {i₁ i₂ i₃} f g := extendToSucc.map_comp hj F τ i₁ i₂ i₃ (leOfHom f) (leOfHom g) i₃.2

/-- The isomorphism `(extendToSucc hj F τ).obj ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j` -/
def extendToSuccObjIso (i : Set.Iic j) :
    (extendToSucc hj F τ).obj ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i :=
  extendToSucc.objIso F X i


at h₂
      rw [map_eq hj F τ i₁ i₂ _ h₂]
      dsimp [map]
      rw [dif_neg h₁, dif_pos (h₁₂.trans h₂), dif_neg h₁, dif_pos h₂,
        assoc, assoc, Iso.inv_hom_id_assoc,comp_id, ← map_comp_assoc, homOfLE_comp]
    · rw [map_id, comp_id]

end extendToSucc

open extendToSucc in
include hj in
/-- The extension to `Set.Iic (Order.succ j) ⥤ C` of a functor `F : Set.Iic j ⥤ C`,
when we specify a morphism `F.obj ⟨j, _⟩ ⟶ X`. -/
def extendToSucc : Set.Iic (Order.succ j) ⥤ C where
  obj := obj F X
  map {i₁ i₂} f := map hj F τ i₁ i₂ (leOfHom f) i₂.2
  map_id _ := extendToSucc.map_id _ F τ _ _
  map_comp {i₁ i₂ i₃} f g := extendToSucc.map_comp hj F τ i₁ i₂ i₃ (leOfHom f) (leOfHom g) i₃.2

/-- The isomorphism `(extendToSucc hj F τ).obj ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j` -/
def extendToSuccObjIso (i : Set.Iic j) :
    (extendToSucc hj F τ).obj ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i :=
  extendToSucc.objIso F X i

/-- The isomorphism `(extendToSucc hj F τ).obj ⟨Order.succ j, _⟩ ≅ X`. -/
def extendToSuccObjSuccIso :
                                                /-
                                                  C : Type u_1
                                                  inst✝² : CategoryTheory.Category.{?u.48243, u_1} C
                                                  J : Type u
                                                  inst✝¹ : LinearOrder J
                                                  inst✝ : SuccOrder J
                                                  j : J
                                                  hj : Not (IsMax j)
                                                  F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                  X : C
                                                  τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                                                  ⊢ Membership.mem (Set.Iic (Order.succ j)) (Order.succ j)
                                                -/
    (extendToSucc hj F τ).obj ⟨Order.succ j, by simp⟩ ≅ X :=
                                                /-
                                                  🎉 no goals
                                                -/
  extendToSucc.objSuccIso hj F X


omp_assoc, homOfLE_comp]
    · rw [map_id, comp_id]

end extendToSucc

open extendToSucc in
include hj in
/-- The extension to `Set.Iic (Order.succ j) ⥤ C` of a functor `F : Set.Iic j ⥤ C`,
when we specify a morphism `F.obj ⟨j, _⟩ ⟶ X`. -/
def extendToSucc : Set.Iic (Order.succ j) ⥤ C where
  obj := obj F X
  map {i₁ i₂} f := map hj F τ i₁ i₂ (leOfHom f) i₂.2
  map_id _ := extendToSucc.map_id _ F τ _ _
  map_comp {i₁ i₂ i₃} f g := extendToSucc.map_comp hj F τ i₁ i₂ i₃ (leOfHom f) (leOfHom g) i₃.2

/-- The isomorphism `(extendToSucc hj F τ).obj ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j` -/
def extendToSuccObjIso (i : Set.Iic j) :
    (extendToSucc hj F τ).obj ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i :=
  extendToSucc.objIso F X i

/-- The isomorphism `(extendToSucc hj F τ).obj ⟨Order.succ j, _⟩ ≅ X`. -/
def extendToSuccObjSuccIso :
    (extendToSucc hj F τ).obj ⟨Order.succ j, by simp⟩ ≅ X :=
  extendToSucc.objSuccIso hj F X

@[reassoc]
lemma extendToSuccObjIso_hom_naturality (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    (extendToSucc hj F τ).map (homOfLE hi :
      ⟨i₁, hi.trans (hi₂.trans (Order.le_succ j))⟩ ⟶ ⟨i₂, hi₂.trans (Order.le_succ j)⟩) ≫
    (extendToSuccObjIso hj F τ ⟨i₂, hi₂⟩).hom =
      (extendToSuccObjIso hj F τ ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    i₁ i₂ : J
    hi : LE.le i₁ i₂
    hi₂ : LE.le i₂ j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.extendToSucc …
  -/
  dsimp [extendToSucc, extendToSuccObjIso]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    i₁ i₂ : J
    hi : LE.le i₁ i₂
    hi₂ : LE.le i₂ j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.extendToSucc. …
  -/
  rw [extendToSucc.map_eq _ _ _ _ _ _ hi₂, assoc, assoc, Iso.inv_hom_id, comp_id]
  /-
    🎉 no goals
  -/


 The isomorphism `(extendToSucc hj F τ).obj ⟨i, _⟩ ≅ F.obj i` when `i : Set.Iic j` -/
def extendToSuccObjIso (i : Set.Iic j) :
    (extendToSucc hj F τ).obj ⟨i, i.2.trans (Order.le_succ j)⟩ ≅ F.obj i :=
  extendToSucc.objIso F X i

/-- The isomorphism `(extendToSucc hj F τ).obj ⟨Order.succ j, _⟩ ≅ X`. -/
def extendToSuccObjSuccIso :
    (extendToSucc hj F τ).obj ⟨Order.succ j, by simp⟩ ≅ X :=
  extendToSucc.objSuccIso hj F X

@[reassoc]
lemma extendToSuccObjIso_hom_naturality (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    (extendToSucc hj F τ).map (homOfLE hi :
      ⟨i₁, hi.trans (hi₂.trans (Order.le_succ j))⟩ ⟶ ⟨i₂, hi₂.trans (Order.le_succ j)⟩) ≫
    (extendToSuccObjIso hj F τ ⟨i₂, hi₂⟩).hom =
      (extendToSuccObjIso hj F τ ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) := by
  dsimp [extendToSucc, extendToSuccObjIso]
  rw [extendToSucc.map_eq _ _ _ _ _ _ hi₂, assoc, assoc, Iso.inv_hom_id, comp_id]

/-- The isomorphism expressing that `extendToSucc hj F τ` extends `F`. -/
@[simps!]
def extendToSuccRestrictionLEIso :
    Iteration.restrictionLE (extendToSucc hj F τ) (Order.le_succ j) ≅ F :=
  NatIso.ofComponents (extendToSuccObjIso hj F τ) (by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.56239, u_1} C
      J : Type u
      inst✝¹ : LinearOrder J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      F : CategoryTheory.Functor (↑(Set.Iic j)) C
      X : C
      τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
      ⊢ ∀ {X_1 Y : ↑(Set.Iic j)} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.Category …
    -/
    rintro ⟨i₁, h₁⟩ ⟨i₂, h₂⟩ f
    /-
      case mk.mk
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.56239, u_1} C
      J : Type u
      inst✝¹ : LinearOrder J
      inst✝ : SuccOrder J
      j : J
      hj : Not (IsMax j)
      F : CategoryTheory.Functor (↑(Set.Iic j)) C
      X : C
      τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
      i₁ : J
      h₁ : Membership.mem (Set.Iic j) i₁
      i₂ : J
      h₂ : Membership.mem (Set.Iic j) i₂
      f : Quiver.Hom ⟨i₁, h₁⟩ ⟨i₂, h₂⟩
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.Iteration.re …
    -/
    apply extendToSuccObjIso_hom_naturality)
    /-
      🎉 no goals
    -/


SuccIso :
    (extendToSucc hj F τ).obj ⟨Order.succ j, by simp⟩ ≅ X :=
  extendToSucc.objSuccIso hj F X

@[reassoc]
lemma extendToSuccObjIso_hom_naturality (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    (extendToSucc hj F τ).map (homOfLE hi :
      ⟨i₁, hi.trans (hi₂.trans (Order.le_succ j))⟩ ⟶ ⟨i₂, hi₂.trans (Order.le_succ j)⟩) ≫
    (extendToSuccObjIso hj F τ ⟨i₂, hi₂⟩).hom =
      (extendToSuccObjIso hj F τ ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) := by
  dsimp [extendToSucc, extendToSuccObjIso]
  rw [extendToSucc.map_eq _ _ _ _ _ _ hi₂, assoc, assoc, Iso.inv_hom_id, comp_id]

/-- The isomorphism expressing that `extendToSucc hj F τ` extends `F`. -/
@[simps!]
def extendToSuccRestrictionLEIso :
    Iteration.restrictionLE (extendToSucc hj F τ) (Order.le_succ j) ≅ F :=
  NatIso.ofComponents (extendToSuccObjIso hj F τ) (by
    rintro ⟨i₁, h₁⟩ ⟨i₂, h₂⟩ f
    apply extendToSuccObjIso_hom_naturality)

lemma extentToSucc_map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    (extendToSucc hj F τ).map (homOfLE hi :
      ⟨i₁, hi.trans (hi₂.trans (Order.le_succ j))⟩ ⟶ ⟨i₂, hi₂.trans (Order.le_succ j)⟩) =
      (extendToSuccObjIso hj F τ ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) ≫
      (extendToSuccObjIso hj F τ ⟨i₂, hi₂⟩).inv := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    J : Type u
    inst✝¹ : LinearOrder J
    inst✝ : SuccOrder J
    j : J
    hj : Not (IsMax j)
    F : CategoryTheory.Functor (↑(Set.Iic j)) C
    X : C
    τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
    i₁ i₂ : J
    hi : LE.le i₁ i₂
    hi₂ : LE.le i₂ j
    ⊢ Eq ((CategoryTheory.Functor.extendToSucc hj F τ).map (CategoryTheory.homOfLE …
  -/
  rw [← extendToSuccObjIso_hom_naturality_assoc, Iso.hom_inv_id, comp_id]
  /-
    🎉 no goals
  -/


⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) := by
  dsimp [extendToSucc, extendToSuccObjIso]
  rw [extendToSucc.map_eq _ _ _ _ _ _ hi₂, assoc, assoc, Iso.inv_hom_id, comp_id]

/-- The isomorphism expressing that `extendToSucc hj F τ` extends `F`. -/
@[simps!]
def extendToSuccRestrictionLEIso :
    Iteration.restrictionLE (extendToSucc hj F τ) (Order.le_succ j) ≅ F :=
  NatIso.ofComponents (extendToSuccObjIso hj F τ) (by
    rintro ⟨i₁, h₁⟩ ⟨i₂, h₂⟩ f
    apply extendToSuccObjIso_hom_naturality)

lemma extentToSucc_map (i₁ i₂ : J) (hi : i₁ ≤ i₂) (hi₂ : i₂ ≤ j) :
    (extendToSucc hj F τ).map (homOfLE hi :
      ⟨i₁, hi.trans (hi₂.trans (Order.le_succ j))⟩ ⟶ ⟨i₂, hi₂.trans (Order.le_succ j)⟩) =
      (extendToSuccObjIso hj F τ ⟨i₁, hi.trans hi₂⟩).hom ≫ F.map (homOfLE hi) ≫
      (extendToSuccObjIso hj F τ ⟨i₂, hi₂⟩).inv := by
  rw [← extendToSuccObjIso_hom_naturality_assoc, Iso.hom_inv_id, comp_id]

lemma extendToSucc_map_le_succ :
    (extendToSucc hj F τ).map (homOfLE (Order.le_succ j)) =
                                          /-
                                            C : Type u_1
                                            inst✝² : CategoryTheory.Category.{?u.63484, u_1} C
                                            J : Type u
                                            inst✝¹ : LinearOrder J
                                            inst✝ : SuccOrder J
                                            j : J
                                            hj : Not (IsMax j)
                                            F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                            X : C
                                            τ : Quiver.Hom (F.obj ⟨j, ⋯⟩) X
                                            ⊢ Membership.mem (Set.Iic j) j
                                          -/
        (extendToSuccObjIso hj F τ ⟨j, by simp⟩).hom ≫ τ ≫
                                          /-
                                            🎉 no goals
                                          -/
          (extendToSuccObjSuccIso hj F τ).inv :=
  extendToSucc.map_self_succ _ _ _


