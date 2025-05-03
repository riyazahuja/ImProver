/-- The map `F.obj ⟨i, _⟩ ⟶ F.obj ⟨Order.succ i, _⟩` when `F : Set.Iic j ⥤ C`
and `i : J` is such that `i < j`. -/
noncomputable abbrev mapSucc' [SuccOrder J] (i : J) (hi : i < j) :
    F.obj ⟨i, hi.le⟩ ⟶ F.obj ⟨Order.succ i, Order.succ_le_of_lt hi⟩ :=
  F.map <| homOfLE <| Subtype.mk_le_mk.2 <| Order.le_succ i


/-- The functor `Set.Iio i ⥤ C` obtained by "restriction" of `F : Set.Iic j ⥤ C`
when `i ≤ j`. -/
def restrictionLT : Set.Iio i ⥤ C :=
  (monotone_inclusion_lt_le_of_le hi).functor ⋙ F


@[simp]
lemma restrictionLT_obj (k : J) (hk : k < i) :
    (restrictionLT F hi).obj ⟨k, hk⟩ = F.obj ⟨k, hk.le.trans hi⟩ := rfl


@[simp]
lemma restrictionLT_map {k₁ k₂ : Set.Iio i} (φ : k₁ ⟶ k₂) :
                                                    /-
                                                      C : Type u_1
                                                      inst✝¹ : CategoryTheory.Category.{?u.3012, u_1} C
                                                      Φ : CategoryTheory.Functor C C
                                                      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                      J : Type u
                                                      inst✝ : Preorder J
                                                      j : J
                                                      F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                      i : J
                                                      hi : LE.le i j
                                                      k₁ k₂ : ↑(Set.Iio i)
                                                      φ : Quiver.Hom k₁ k₂
                                                      ⊢ LE.le (⋯.functor.obj k₁) (⋯.functor.obj k₂)
                                                    -/
    (restrictionLT F hi).map φ = F.map (homOfLE (by simpa using leOfHom φ)) := rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Given `F : Set.Iic j ⥤ C`, `i : J` such that `hi : i ≤ j`, this is the
cocone consisting of all maps `F.obj ⟨k, hk⟩ ⟶ F.obj ⟨i, hi⟩` for `k : J` such that `k < i`. -/
@[simps]
def coconeOfLE : Cocone (restrictionLT F hi) where
  pt := F.obj ⟨i, hi⟩
  ι :=
                                               /-
                                                 C : Type u_1
                                                 inst✝¹ : CategoryTheory.Category.{?u.5166, u_1} C
                                                 Φ : CategoryTheory.Functor C C
                                                 ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                 J : Type u
                                                 inst✝ : Preorder J
                                                 j : J
                                                 F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                 i : J
                                                 hi : LE.le i j
                                                 x✝ : ↑(Set.Iio i)
                                                 k : J
                                                 hk : Membership.mem (Set.Iio i) k
                                                 ⊢ LE.le (⋯.functor.obj ⟨k, hk⟩) ⟨i, hi⟩
                                               -/
    { app := fun ⟨k, hk⟩ => F.map (homOfLE (by simpa using hk.le))
                                               /-
                                                 🎉 no goals
                                               -/
      naturality := fun ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩ _ => by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.5166, u_1} C
          Φ : CategoryTheory.Functor C C
          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
          J : Type u
          inst✝ : Preorder J
          j : J
          F : CategoryTheory.Functor (↑(Set.Iic j)) C
          i : J
          hi : LE.le i j
          x✝² x✝¹ : ↑(Set.Iio i)
          k₁ : J
          hk₁ : Membership.mem (Set.Iio i) k₁
          k₂ : J
          hk₂ : Membership.mem (Set.Iio i) k₂
          x✝ : Quiver.Hom ⟨k₁, hk₁⟩ ⟨k₂, hk₂⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.Iteration.re …
        -/
        simp [comp_id, ← Functor.map_comp, homOfLE_comp] }
        /-
          🎉 no goals
        -/


/-- The functor `Set.Iic i ⥤ C` obtained by "restriction" of `F : Set.Iic j ⥤ C`
when `i ≤ j`. -/
def restrictionLE : Set.Iic i ⥤ C :=
  (monotone_inclusion_le_le_of_le hi).functor ⋙ F


@[simp]
lemma restrictionLE_obj (k : J) (hk : k ≤ i) :
    (restrictionLE F hi).obj ⟨k, hk⟩ = F.obj ⟨k, hk.trans hi⟩ := rfl


@[simp]
lemma restrictionLE_map {k₁ k₂ : Set.Iic i} (φ : k₁ ⟶ k₂) :
                                                    /-
                                                      C : Type u_1
                                                      inst✝¹ : CategoryTheory.Category.{?u.10970, u_1} C
                                                      Φ : CategoryTheory.Functor C C
                                                      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                      J : Type u
                                                      inst✝ : Preorder J
                                                      j : J
                                                      F : CategoryTheory.Functor (↑(Set.Iic j)) C
                                                      i : J
                                                      hi : LE.le i j
                                                      k₁ k₂ : ↑(Set.Iic i)
                                                      φ : Quiver.Hom k₁ k₂
                                                      ⊢ LE.le (⋯.functor.obj k₁) (⋯.functor.obj k₂)
                                                    -/
    (restrictionLE F hi).map φ = F.map (homOfLE (by simpa using leOfHom φ)) := rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The category of `j`th iterations of a functor `Φ` equipped with a natural
transformation `ε : 𝟭 C ⟶ Φ`. An object consists of the data of all iterations
of `Φ` for `i : J` such that `i ≤ j` (this is the field `F`). Such objects are
equipped with data and properties which characterizes the iterations up to a unique
isomorphism for the three types of elements: `⊥`, successors, limit elements. -/
structure Iteration [Preorder J] [OrderBot J] [SuccOrder J] (j : J) where
  /-- The data of all `i`th iterations for `i : J` such that `i ≤ j`. -/
  F : Set.Iic j ⥤ C ⥤ C
  /-- The zeroth iteration is the identity functor. -/
  isoZero : F.obj ⟨⊥, bot_le⟩ ≅ 𝟭 C
  /-- The iteration on a successor element is obtained by composition of
  the previous iteration with `Φ`. -/
  isoSucc (i : J) (hi : i < j) : F.obj ⟨Order.succ i, Order.succ_le_of_lt hi⟩ ≅ F.obj ⟨i, hi.le⟩ ⋙ Φ
  /-- The natural map from an iteration to its successor is induced by `ε`. -/
  mapSucc'_eq (i : J) (hi : i < j) :
    Iteration.mapSucc' F i hi = whiskerLeft _ ε ≫ (isoSucc i hi).inv
  /-- If `i` is a limit element, the `i`th iteration is the colimit
  of `k`th iterations for `k < i`. -/
  isColimit (i : J) (hi : Order.IsSuccLimit i) (hij : i ≤ j) :
    IsColimit (Iteration.coconeOfLE F hij)


/-- For `iter : Φ.Iteration.ε j`, this is the map
`iter.F.obj ⟨i, _⟩ ⟶ iter.F.obj ⟨Order.succ i, _⟩` if `i : J` is such that `i < j`. -/
noncomputable abbrev mapSucc (i : J) (hi : i < j) :
    iter.F.obj ⟨i, hi.le⟩ ⟶ iter.F.obj ⟨Order.succ i, Order.succ_le_of_lt hi⟩ :=
  mapSucc' iter.F i hi


lemma mapSucc_eq (i : J) (hi : i < j) :
    iter.mapSucc i hi = whiskerLeft _ ε ≫ (iter.isoSucc i hi).inv :=
  iter.mapSucc'_eq _ hi


/-- A morphism between two objects `iter₁` and `iter₂` in the
category `Φ.Iteration ε j` of `j`th iterations of a functor `Φ`
equipped with a natural transformation `ε : 𝟭 C ⟶ Φ` consists of a natural
transformation `natTrans : iter₁.F ⟶ iter₂.F` which is compatible with the
isomorphisms `isoZero` and `isoSucc`. -/
structure Hom where
  /-- A natural transformation `iter₁.F ⟶ iter₂.F` -/
  natTrans : iter₁.F ⟶ iter₂.F
  natTrans_app_zero :
    natTrans.app ⟨⊥, bot_le⟩ = iter₁.isoZero.hom ≫ iter₂.isoZero.inv := by aesop_cat
  natTrans_app_succ (i : J) (hi : i < j) :
    natTrans.app ⟨Order.succ i, Order.succ_le_of_lt hi⟩ = (iter₁.isoSucc i hi).hom ≫
      whiskerRight (natTrans.app ⟨i, hi.le⟩) _ ≫ (iter₂.isoSucc i hi).inv := by aesop_cat


attribute [simp, reassoc] natTrans_app_zero


/-- The identity morphism in the category `Φ.Iteration ε j`. -/
@[simps]
def id : Hom iter₁ iter₁ where
  natTrans := 𝟙 _


lemma ext' {f g : Hom iter₁ iter₂} (h : f.natTrans = g.natTrans) : f = g := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝² : Preorder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    f g : iter₁.Hom iter₂
    h : Eq f.natTrans g.natTrans
    ⊢ Eq f g
  -/
  cases f
  /-
    case mk
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝² : Preorder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    g : iter₁.Hom iter₂
    natTrans✝ : Quiver.Hom iter₁.F iter₂.F
    natTrans_app_zero✝ : Eq (natTrans✝.app ⟨Bot.bot, ⋯⟩) (CategoryTheory.CategoryS …
    natTrans_app_succ✝ : ∀ (i : J) (hi : LT.lt i j), Eq (natTrans✝.app ⟨Order.succ …
    h : Eq { natTrans := natTrans✝, natTrans_app_zero := natTrans_app_zero✝, natTr …
    ⊢ Eq { natTrans := natTrans✝, natTrans_app_zero := natTrans_app_zero✝, natTran …
  -/
  cases g
  /-
    case mk.mk
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝² : Preorder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    natTrans✝¹ : Quiver.Hom iter₁.F iter₂.F
    natTrans_app_zero✝¹ : Eq (natTrans✝¹.app ⟨Bot.bot, ⋯⟩) (CategoryTheory.Categor …
    natTrans_app_succ✝¹ : ∀ (i : J) (hi : LT.lt i j), Eq (natTrans✝¹.app ⟨Order.su …
    natTrans✝ : Quiver.Hom iter₁.F iter₂.F
    natTrans_app_zero✝ : Eq (natTrans✝.app ⟨Bot.bot, ⋯⟩) (CategoryTheory.CategoryS …
    natTrans_app_succ✝ : ∀ (i : J) (hi : LT.lt i j), Eq (natTrans✝.app ⟨Order.succ …
    h : Eq { natTrans := natTrans✝¹, natTrans_app_zero := natTrans_app_zero✝¹, nat …
    ⊢ Eq { natTrans := natTrans✝¹, natTrans_app_zero := natTrans_app_zero✝¹, natTr …
  -/
  subst h
  /-
    case mk.mk
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝² : Preorder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    natTrans✝ : Quiver.Hom iter₁.F iter₂.F
    natTrans_app_zero✝¹ : Eq (natTrans✝.app ⟨Bot.bot, ⋯⟩) (CategoryTheory.Category …
    natTrans_app_succ✝¹ : ∀ (i : J) (hi : LT.lt i j), Eq (natTrans✝.app ⟨Order.suc …
    natTrans_app_zero✝ : Eq ({ natTrans := natTrans✝, natTrans_app_zero := natTran …
    natTrans_app_succ✝ : ∀ (i : J) (hi : LT.lt i j), Eq ({ natTrans := natTrans✝,  …
    ⊢ Eq { natTrans := natTrans✝, natTrans_app_zero := natTrans_app_zero✝¹, natTra …
  -/
  rfl
  /-
    🎉 no goals
  -/


attribute [local ext] ext'


/-- The composition of morphisms in the category `Iteration ε j`. -/
@[simps]
def comp {iter₃ : Iteration ε j} (f : Hom iter₁ iter₂) (g : Hom iter₂ iter₃) :
    Hom iter₁ iter₃ where
  natTrans := f.natTrans ≫ g.natTrans
                               /-
                                 C : Type u_1
                                 inst✝³ : CategoryTheory.Category.{?u.23542, u_1} C
                                 Φ : CategoryTheory.Functor C C
                                 ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                 J : Type u
                                 j : J
                                 inst✝² : Preorder J
                                 inst✝¹ : OrderBot J
                                 inst✝ : SuccOrder J
                                 iter₁ iter₂ iter₃ : CategoryTheory.Functor.Iteration ε j
                                 f : iter₁.Hom iter₂
                                 g : iter₂.Hom iter₃
                                 i : J
                                 hi : LT.lt i j
                                 ⊢ Eq ((CategoryTheory.CategoryStruct.comp f.natTrans g.natTrans).app ⟨Order.su …
                               -/
  natTrans_app_succ i hi := by simp [natTrans_app_succ _ _ hi]
                               /-
                                 🎉 no goals
                               -/


instance : Category (Iteration ε j) where
  Hom := Hom
  id := id
  comp := comp


instance {J} {j : J} [PartialOrder J] [OrderBot J] [WellFoundedLT J] [SuccOrder J]
    {iter₁ iter₂ : Iteration ε j} :
    Subsingleton (iter₁ ⟶ iter₂) where
  allEq f g := by
    /-
      C : Type u_1
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J✝ : Type u
      j✝ : J✝
      inst✝⁶ : Preorder J✝
      inst✝⁵ : OrderBot J✝
      inst✝⁴ : SuccOrder J✝
      iter₁✝ iter₂✝ : CategoryTheory.Functor.Iteration ε j✝
      J : Type u_2
      j : J
      inst✝³ : PartialOrder J
      inst✝² : OrderBot J
      inst✝¹ : WellFoundedLT J
      inst✝ : SuccOrder J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      f g : Quiver.Hom iter₁ iter₂
      ⊢ Eq f g
    -/
    apply ext'
    suffices ∀ i hi, f.natTrans.app ⟨i, hi⟩ = g.natTrans.app ⟨i, hi⟩ by
      ext ⟨i, hi⟩ : 2
      apply this
    /-
      case h
      C : Type u_1
      inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
      Φ : CategoryTheory.Functor C C
      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
      J✝ : Type u
      j✝ : J✝
      inst✝⁶ : Preorder J✝
      inst✝⁵ : OrderBot J✝
      inst✝⁴ : SuccOrder J✝
      iter₁✝ iter₂✝ : CategoryTheory.Functor.Iteration ε j✝
      J : Type u_2
      j : J
      inst✝³ : PartialOrder J
      inst✝² : OrderBot J
      inst✝¹ : WellFoundedLT J
      inst✝ : SuccOrder J
      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
      f g : Quiver.Hom iter₁ iter₂
      ⊢ ∀ (i : J) (hi : Membership.mem (Set.Iic j) i), Eq (f.natTrans.app ⟨i, hi⟩) ( …
    -/
    intro i
    induction i using SuccOrder.limitRecOn with
    | hm j H =>
      obtain rfl := H.eq_bot
      simp [natTrans_app_zero]
    | hs j H IH =>
      intro hj
      simp [Hom.natTrans_app_succ, IH, (Order.lt_succ_of_not_isMax H).trans_le hj]
    | hl j H IH =>
      refine fun hj ↦ (iter₁.isColimit j H hj).hom_ext ?_
      rintro ⟨k, hk⟩
      simp [IH k hk]


@[simp]
lemma natTrans_id : Hom.natTrans (𝟙 iter₁) = 𝟙 _ := rfl


@[simp, reassoc]
lemma natTrans_comp {iter₃ : Iteration ε j} (φ : iter₁ ⟶ iter₂) (ψ : iter₂ ⟶ iter₃) :
    (φ ≫ ψ).natTrans = φ.natTrans ≫ ψ.natTrans := rfl


@[reassoc]
lemma natTrans_naturality (φ : iter₁ ⟶ iter₂) (i₁ i₂ : J) (h : i₁ ≤ i₂) (h' : i₂ ≤ j) :
                    /-
                      C : Type u_1
                      inst✝³ : CategoryTheory.Category.{?u.36558, u_1} C
                      Φ : CategoryTheory.Functor C C
                      ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                      J : Type u
                      j : J
                      inst✝² : Preorder J
                      inst✝¹ : OrderBot J
                      inst✝ : SuccOrder J
                      iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                      φ : Quiver.Hom iter₁ iter₂
                      i₁ i₂ : J
                      h : LE.le i₁ i₂
                      h' : LE.le i₂ j
                      ⊢ Quiver.Hom ⟨i₁, ⋯⟩ ⟨i₂, h'⟩
                    -/
    iter₁.F.map (by exact homOfLE h) ≫ φ.natTrans.app ⟨i₂, h'⟩ =
                    /-
                      🎉 no goals
                    -/
                                                        /-
                                                          C : Type u_1
                                                          inst✝³ : CategoryTheory.Category.{?u.36558, u_1} C
                                                          Φ : CategoryTheory.Functor C C
                                                          ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
                                                          J : Type u
                                                          j : J
                                                          inst✝² : Preorder J
                                                          inst✝¹ : OrderBot J
                                                          inst✝ : SuccOrder J
                                                          iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
                                                          φ : Quiver.Hom iter₁ iter₂
                                                          i₁ i₂ : J
                                                          h : LE.le i₁ i₂
                                                          h' : LE.le i₂ j
                                                          ⊢ Quiver.Hom ⟨i₁, ⋯⟩ ⟨i₂, h'⟩
                                                        -/
      φ.natTrans.app ⟨i₁, h.trans h'⟩ ≫ iter₂.F.map (by exact homOfLE h) := by
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝² : Preorder J
    inst✝¹ : OrderBot J
    inst✝ : SuccOrder J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    φ : Quiver.Hom iter₁ iter₂
    i₁ i₂ : J
    h : LE.le i₁ i₂
    h' : LE.le i₂ j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (iter₁.F.map (CategoryTheory.homOfLE  …
  -/
  apply φ.natTrans.naturality
  /-
    🎉 no goals
  -/


variable (ε) in
/-- The evaluation functor `Iteration ε j ⥤ C ⥤ C` at `i : J` when `i ≤ j`. -/
@[simps]
def eval {i : J} (hi : i ≤ j) : Iteration ε j ⥤ C ⥤ C where
  obj iter := iter.F.obj ⟨i, hi⟩
  map φ := φ.natTrans.app _


/-- Given `iter : Iteration ε j` and `i : J` such that `i ≤ j`, this is the
induced element in `Iteration ε i`. -/
@[simps F isoZero isoSucc]
def trunc (iter : Iteration ε j) {i : J} (hi : i ≤ j) : Iteration ε i where
  F := restrictionLE iter.F hi
  isoZero := iter.isoZero
  isoSucc k hk := iter.isoSucc k (lt_of_lt_of_le hk hi)
  mapSucc'_eq k hk := iter.mapSucc'_eq k (lt_of_lt_of_le hk hi)
  isColimit k hk' hk := iter.isColimit k hk' (hk.trans hi)


variable (ε) in
/-- The truncation functor `Iteration ε j ⥤ Iteration ε i` when `i ≤ j`. -/
@[simps obj]
def truncFunctor {i : J} (hi : i ≤ j) : Iteration ε j ⥤ Iteration ε i where
  obj iter := iter.trunc hi
  map {iter₁ iter₂} φ :=
    { natTrans := whiskerLeft _ φ.natTrans
      natTrans_app_succ := fun k hk => φ.natTrans_app_succ k (lt_of_lt_of_le hk hi) }


@[simp]
lemma truncFunctor_map_natTrans_app
    (φ : iter₁ ⟶ iter₂) {i : J} (hi : i ≤ j) (k : J) (hk : k ≤ i) :
    ((truncFunctor ε hi).map φ).natTrans.app ⟨k, hk⟩ =
      φ.natTrans.app ⟨k, hk.trans hi⟩ := rfl


lemma congr_app (φ φ' : iter₁ ⟶ iter₂) (i : J) (hi : i ≤ j) :
    φ.natTrans.app ⟨i, hi⟩ = φ'.natTrans.app ⟨i, hi⟩ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝³ : PartialOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    φ φ' : Quiver.Hom iter₁ iter₂
    i : J
    hi : LE.le i j
    ⊢ Eq (φ.natTrans.app ⟨i, hi⟩) (φ'.natTrans.app ⟨i, hi⟩)
  -/
  obtain rfl := Subsingleton.elim φ φ'
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    Φ : CategoryTheory.Functor C C
    ε : Quiver.Hom (CategoryTheory.Functor.id C) Φ
    J : Type u
    j : J
    inst✝³ : PartialOrder J
    inst✝² : OrderBot J
    inst✝¹ : SuccOrder J
    inst✝ : WellFoundedLT J
    iter₁ iter₂ : CategoryTheory.Functor.Iteration ε j
    φ : Quiver.Hom iter₁ iter₂
    i : J
    hi : LE.le i j
    ⊢ Eq (φ.natTrans.app ⟨i, hi⟩) (φ.natTrans.app ⟨i, hi⟩)
  -/
  rfl
  /-
    🎉 no goals
  -/


