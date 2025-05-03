/-- A `2`-square consists of a natural transformation `T ⋙ R ⟶ L ⋙ B`
involving fours functors `T`, `L`, `R`, `B` that are on the
top/left/right/bottom sides of a square of categories. -/
def TwoSquare := T ⋙ R ⟶ L ⋙ B


/-- Constructor for `TwoSquare`. -/
abbrev mk (α : T ⋙ R ⟶ L ⋙ B) : TwoSquare T L R B := α


@[ext]
lemma ext (w w' : TwoSquare T L R B) (h : ∀ (X : C₁), w.app X = w'.app X) :
    w = w' :=
  NatTrans.ext (funext h)


/-- Given `w : TwoSquare T L R B` and `X₃ : C₃`, this is the obvious functor
`CostructuredArrow L X₃ ⥤ CostructuredArrow R (B.obj X₃)`. -/
@[simps! obj map]
def costructuredArrowRightwards (X₃ : C₃) :
    CostructuredArrow L X₃ ⥤ CostructuredArrow R (B.obj X₃) :=
  CostructuredArrow.post L B X₃ ⋙ Comma.mapLeft _ w ⋙
    CostructuredArrow.pre T R (B.obj X₃)


/-- Given `w : TwoSquare T L R B` and `X₂ : C₂`, this is the obvious functor
`StructuredArrow X₂ T ⥤ StructuredArrow (R.obj X₂) B`. -/
@[simps! obj map]
def structuredArrowDownwards (X₂ : C₂) :
    StructuredArrow X₂ T ⥤ StructuredArrow (R.obj X₂) B :=
  StructuredArrow.post X₂ T R ⋙ Comma.mapRight _ w ⋙
    StructuredArrow.pre (R.obj X₂) L B


/-- Given `w : TwoSquare T L R B` and a morphism `g : R.obj X₂ ⟶ B.obj X₃`, this is the
category `StructuredArrow (CostructuredArrow.mk g) (w.costructuredArrowRightwards X₃)`,
see the constructor `StructuredArrowRightwards.mk` for the data that is involved. -/
abbrev StructuredArrowRightwards :=
  StructuredArrow (CostructuredArrow.mk g) (w.costructuredArrowRightwards X₃)


/-- Given `w : TwoSquare T L R B` and a morphism `g : R.obj X₂ ⟶ B.obj X₃`, this is the
category `CostructuredArrow (w.structuredArrowDownwards X₂) (StructuredArrow.mk g)`,
see the constructor `CostructuredArrowDownwards.mk` for the data that is involved. -/
abbrev CostructuredArrowDownwards :=
  CostructuredArrow (w.structuredArrowDownwards X₂) (StructuredArrow.mk g)


/-- Constructor for objects in `w.StructuredArrowRightwards g`. -/
abbrev StructuredArrowRightwards.mk (comm : R.map a ≫ w.app X₁ ≫ B.map b = g) :
    w.StructuredArrowRightwards g :=
  StructuredArrow.mk (Y := CostructuredArrow.mk b) (CostructuredArrow.homMk a comm)


/-- Constructor for objects in `w.CostructuredArrowDownwards g`. -/
abbrev CostructuredArrowDownwards.mk (comm : R.map a ≫ w.app X₁ ≫ B.map b = g) :
    w.CostructuredArrowDownwards g :=
  CostructuredArrow.mk (Y := StructuredArrow.mk a)
                                 /-
                                   C₁ : Type u₁
                                   C₂ : Type u₂
                                   C₃ : Type u₃
                                   C₄ : Type u₄
                                   inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
                                   inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
                                   inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
                                   inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
                                   T : CategoryTheory.Functor C₁ C₂
                                   L : CategoryTheory.Functor C₁ C₃
                                   R : CategoryTheory.Functor C₂ C₄
                                   B : CategoryTheory.Functor C₃ C₄
                                   w : CategoryTheory.TwoSquare T L R B
                                   X₂ : C₂
                                   X₃ : C₃
                                   g : Quiver.Hom (R.obj X₂) (B.obj X₃)
                                   X₁ : C₁
                                   a : Quiver.Hom X₂ (T.obj X₁)
                                   b : Quiver.Hom (L.obj X₁) X₃
                                   comm : Eq (CategoryTheory.CategoryStruct.comp (R.map a) (CategoryTheory.Catego …
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((w.structuredArrowDownwards X₂).obj  …
                                 -/
    (StructuredArrow.homMk b (by simpa using comm))
                                 /-
                                   🎉 no goals
                                 -/


lemma StructuredArrowRightwards.mk_surjective
    (f : w.StructuredArrowRightwards g) :
    ∃ (X₁ : C₁) (a : X₂ ⟶ T.obj X₁) (b : L.obj X₁ ⟶ X₃)
      (comm : R.map a ≫ w.app X₁ ≫ B.map b = g), f = mk w g X₁ a b comm := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g : Quiver.Hom (R.obj X₂) (B.obj X₃)
    f : w.StructuredArrowRightwards g
    ⊢ Exists fun X₁ => Exists fun a => Exists fun b => Exists fun comm => Eq f (Ca …
  -/
  obtain ⟨g, φ, rfl⟩ := StructuredArrow.mk_surjective f
  /-
    case intro.intro
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g✝ : Quiver.Hom (R.obj X₂) (B.obj X₃)
    g : CategoryTheory.CostructuredArrow L X₃
    φ : Quiver.Hom (CategoryTheory.CostructuredArrow.mk g✝) ((w.costructuredArrowR …
    ⊢ Exists fun X₁ => Exists fun a => Exists fun b => Exists fun comm => Eq (Cate …
  -/
  obtain ⟨X₁, b, rfl⟩ := g.mk_surjective
  /-
    case intro.intro.intro.intro
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g : Quiver.Hom (R.obj X₂) (B.obj X₃)
    X₁ : C₁
    b : Quiver.Hom (L.obj X₁) X₃
    φ : Quiver.Hom (CategoryTheory.CostructuredArrow.mk g) ((w.costructuredArrowRi …
    ⊢ Exists fun X₁_1 => Exists fun a => Exists fun b_1 => Exists fun comm => Eq ( …
  -/
  obtain ⟨a, ha, rfl⟩ := CostructuredArrow.homMk_surjective φ
  /-
    case intro.intro.intro.intro.intro.intro
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g : Quiver.Hom (R.obj X₂) (B.obj X₃)
    X₁ : C₁
    b : Quiver.Hom (L.obj X₁) X₃
    a : Quiver.Hom (CategoryTheory.CostructuredArrow.mk g).left ((w.costructuredAr …
    ha : Eq (CategoryTheory.CategoryStruct.comp (R.map a) ((w.costructuredArrowRig …
    ⊢ Exists fun X₁_1 => Exists fun a_1 => Exists fun b_1 => Exists fun comm => Eq …
  -/
  exact ⟨X₁, a, b, by simpa using ha, rfl⟩
  /-
    🎉 no goals
  -/


lemma CostructuredArrowDownwards.mk_surjective
    (f : w.CostructuredArrowDownwards g) :
    ∃ (X₁ : C₁) (a : X₂ ⟶ T.obj X₁) (b : L.obj X₁ ⟶ X₃)
      (comm : R.map a ≫ w.app X₁ ≫ B.map b = g), f = mk w g X₁ a b comm := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g : Quiver.Hom (R.obj X₂) (B.obj X₃)
    f : w.CostructuredArrowDownwards g
    ⊢ Exists fun X₁ => Exists fun a => Exists fun b => Exists fun comm => Eq f (Ca …
  -/
  obtain ⟨g, φ, rfl⟩ := CostructuredArrow.mk_surjective f
  /-
    case intro.intro
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g✝ : Quiver.Hom (R.obj X₂) (B.obj X₃)
    g : CategoryTheory.StructuredArrow X₂ T
    φ : Quiver.Hom ((w.structuredArrowDownwards X₂).obj g) (CategoryTheory.Structu …
    ⊢ Exists fun X₁ => Exists fun a => Exists fun b => Exists fun comm => Eq (Cate …
  -/
  obtain ⟨X₁, a, rfl⟩ := g.mk_surjective
  /-
    case intro.intro.intro.intro
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g : Quiver.Hom (R.obj X₂) (B.obj X₃)
    X₁ : C₁
    a : Quiver.Hom X₂ (T.obj X₁)
    φ : Quiver.Hom ((w.structuredArrowDownwards X₂).obj (CategoryTheory.Structured …
    ⊢ Exists fun X₁_1 => Exists fun a_1 => Exists fun b => Exists fun comm => Eq ( …
  -/
  obtain ⟨b, hb, rfl⟩ := StructuredArrow.homMk_surjective φ
  /-
    case intro.intro.intro.intro.intro.intro
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    X₂ : C₂
    X₃ : C₃
    g : Quiver.Hom (R.obj X₂) (B.obj X₃)
    X₁ : C₁
    a : Quiver.Hom X₂ (T.obj X₁)
    b : Quiver.Hom ((w.structuredArrowDownwards X₂).obj (CategoryTheory.Structured …
    hb : Eq (CategoryTheory.CategoryStruct.comp ((w.structuredArrowDownwards X₂).o …
    ⊢ Exists fun X₁_1 => Exists fun a_1 => Exists fun b_1 => Exists fun comm => Eq …
  -/
  exact ⟨X₁, a, b, by simpa using hb, rfl⟩
  /-
    🎉 no goals
  -/


/-- Given `w : TwoSquare T L R B` and a morphism `g : R.obj X₂ ⟶ B.obj X₃`, this is
the obvious functor `w.StructuredArrowRightwards g ⥤ w.CostructuredArrowDownwards g`. -/
@[simps]
def functor : w.StructuredArrowRightwards g ⥤ w.CostructuredArrowDownwards g where
  obj f := CostructuredArrow.mk (Y := StructuredArrow.mk f.hom.left)
                                             /-
                                               C₁ : Type u₁
                                               C₂ : Type u₂
                                               C₃ : Type u₃
                                               C₄ : Type u₄
                                               inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
                                               inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
                                               inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
                                               inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
                                               T : CategoryTheory.Functor C₁ C₂
                                               L : CategoryTheory.Functor C₁ C₃
                                               R : CategoryTheory.Functor C₂ C₄
                                               B : CategoryTheory.Functor C₃ C₄
                                               w : CategoryTheory.TwoSquare T L R B
                                               X₂ : C₂
                                               X₃ : C₃
                                               g : Quiver.Hom (R.obj X₂) (B.obj X₃)
                                               f : w.StructuredArrowRightwards g
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp ((w.structuredArrowDownwards X₂).obj  …
                                             -/
      (StructuredArrow.homMk f.right.hom (by simpa using CostructuredArrow.w f.hom))
                                             /-
                                               🎉 no goals
                                             -/
  map {f₁ f₂} φ :=
    CostructuredArrow.homMk (StructuredArrow.homMk φ.right.left
          /-
            C₁ : Type u₁
            C₂ : Type u₂
            C₃ : Type u₃
            C₄ : Type u₄
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
            inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
            inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
            T : CategoryTheory.Functor C₁ C₂
            L : CategoryTheory.Functor C₁ C₃
            R : CategoryTheory.Functor C₂ C₄
            B : CategoryTheory.Functor C₃ C₄
            w : CategoryTheory.TwoSquare T L R B
            X₂ : C₂
            X₃ : C₃
            g : Quiver.Hom (R.obj X₂) (B.obj X₃)
            f₁ f₂ : w.StructuredArrowRightwards g
            φ : Quiver.Hom f₁ f₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun f => CategoryTheory.Costructure …
          -/
      (by dsimp; rw [← StructuredArrow.w φ]; rfl))
                                             /-
                                               🎉 no goals
                                             -/
          /-
            C₁ : Type u₁
            C₂ : Type u₂
            C₃ : Type u₃
            C₄ : Type u₄
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
            inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
            inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
            T : CategoryTheory.Functor C₁ C₂
            L : CategoryTheory.Functor C₁ C₃
            R : CategoryTheory.Functor C₂ C₄
            B : CategoryTheory.Functor C₃ C₄
            w : CategoryTheory.TwoSquare T L R B
            X₂ : C₂
            X₃ : C₃
            g : Quiver.Hom (R.obj X₂) (B.obj X₃)
            f₁ f₂ : w.StructuredArrowRightwards g
            φ : Quiver.Hom f₁ f₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((w.structuredArrowDownwards X₂).map  …
          -/
      (by ext; exact CostructuredArrow.w φ.right)
               /-
                 🎉 no goals
               -/
  map_id _ := rfl
  map_comp _ _ := rfl


/-- Given `w : TwoSquare T L R B` and a morphism `g : R.obj X₂ ⟶ B.obj X₃`, this is
the obvious functor `w.CostructuredArrowDownwards g ⥤ w.StructuredArrowRightwards g`. -/
@[simps]
def inverse : w.CostructuredArrowDownwards g ⥤ w.StructuredArrowRightwards g where
  obj f := StructuredArrow.mk (Y := CostructuredArrow.mk f.hom.right)
                                              /-
                                                C₁ : Type u₁
                                                C₂ : Type u₂
                                                C₃ : Type u₃
                                                C₄ : Type u₄
                                                inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
                                                inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
                                                inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
                                                inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
                                                T : CategoryTheory.Functor C₁ C₂
                                                L : CategoryTheory.Functor C₁ C₃
                                                R : CategoryTheory.Functor C₂ C₄
                                                B : CategoryTheory.Functor C₃ C₄
                                                w : CategoryTheory.TwoSquare T L R B
                                                X₂ : C₂
                                                X₃ : C₃
                                                g : Quiver.Hom (R.obj X₂) (B.obj X₃)
                                                f : w.CostructuredArrowDownwards g
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map f.left.hom) ((w.costructuredAr …
                                              -/
      (CostructuredArrow.homMk f.left.hom (by simpa using StructuredArrow.w f.hom))
                                              /-
                                                🎉 no goals
                                              -/
  map {f₁ f₂} φ :=
    StructuredArrow.homMk (CostructuredArrow.homMk φ.left.right
          /-
            C₁ : Type u₁
            C₂ : Type u₂
            C₃ : Type u₃
            C₄ : Type u₄
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
            inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
            inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
            T : CategoryTheory.Functor C₁ C₂
            L : CategoryTheory.Functor C₁ C₃
            R : CategoryTheory.Functor C₂ C₄
            B : CategoryTheory.Functor C₃ C₄
            w : CategoryTheory.TwoSquare T L R B
            X₂ : C₂
            X₃ : C₃
            g : Quiver.Hom (R.obj X₂) (B.obj X₃)
            f₁ f₂ : w.CostructuredArrowDownwards g
            φ : Quiver.Hom f₁ f₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map φ.left.right) ((fun f => Categ …
          -/
      (by dsimp; rw [← CostructuredArrow.w φ]; rfl))
                                               /-
                                                 🎉 no goals
                                               -/
          /-
            C₁ : Type u₁
            C₂ : Type u₂
            C₃ : Type u₃
            C₄ : Type u₄
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
            inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
            inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
            inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
            T : CategoryTheory.Functor C₁ C₂
            L : CategoryTheory.Functor C₁ C₃
            R : CategoryTheory.Functor C₂ C₄
            B : CategoryTheory.Functor C₃ C₄
            w : CategoryTheory.TwoSquare T L R B
            X₂ : C₂
            X₃ : C₃
            g : Quiver.Hom (R.obj X₂) (B.obj X₃)
            f₁ f₂ : w.CostructuredArrowDownwards g
            φ : Quiver.Hom f₁ f₂
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun f => CategoryTheory.StructuredA …
          -/
      (by ext; exact StructuredArrow.w φ.left)
               /-
                 🎉 no goals
               -/
  map_id _ := rfl
  map_comp _ _ := rfl


/-- Given `w : TwoSquare T L R B` and a morphism `g : R.obj X₂ ⟶ B.obj X₃`, this is
the obvious equivalence of categories
`w.StructuredArrowRightwards g ≌ w.CostructuredArrowDownwards g`. -/
@[simps functor inverse unitIso counitIso]
def equivalenceJ : w.StructuredArrowRightwards g ≌ w.CostructuredArrowDownwards g where
  functor := EquivalenceJ.functor w g
  inverse := EquivalenceJ.inverse w g
  unitIso := Iso.refl _
  counitIso := Iso.refl _


lemma isConnected_rightwards_iff_downwards :
    IsConnected (w.StructuredArrowRightwards g) ↔ IsConnected (w.CostructuredArrowDownwards g) :=
  isConnected_iff_of_equivalence (w.equivalenceJ g)


/-- The functor `w.CostructuredArrowDownwards g ⥤ w.CostructuredArrowDownwards g'` induced
by a morphism `γ` such that `R.map γ ≫ g = g'`. -/
@[simps]
def costructuredArrowDownwardsPrecomp
    {X₂ X₂' : C₂} {X₃ : C₃} (g : R.obj X₂ ⟶ B.obj X₃) (g' : R.obj X₂' ⟶ B.obj X₃)
    (γ : X₂' ⟶ X₂) (hγ : R.map γ ≫ g = g') :
    w.CostructuredArrowDownwards g ⥤ w.CostructuredArrowDownwards g' where
  obj A := CostructuredArrowDownwards.mk _ _ A.left.right (γ ≫ A.left.hom) A.hom.right
        /-
          C₁ : Type u₁
          C₂ : Type u₂
          C₃ : Type u₃
          C₄ : Type u₄
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
          inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
          inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
          inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
          T : CategoryTheory.Functor C₁ C₂
          L : CategoryTheory.Functor C₁ C₃
          R : CategoryTheory.Functor C₂ C₄
          B : CategoryTheory.Functor C₃ C₄
          w : CategoryTheory.TwoSquare T L R B
          X₂ X₂' : C₂
          X₃ : C₃
          g : Quiver.Hom (R.obj X₂) (B.obj X₃)
          g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
          γ : Quiver.Hom X₂' X₂
          hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
          A : w.CostructuredArrowDownwards g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map (CategoryTheory.CategoryStruct …
        -/
    (by simpa [← hγ] using R.map γ ≫= StructuredArrow.w A.hom)
        /-
          🎉 no goals
        -/
  map {A A'} φ := CostructuredArrow.homMk (StructuredArrow.homMk φ.left.right (by
      /-
        C₁ : Type u₁
        C₂ : Type u₂
        C₃ : Type u₃
        C₄ : Type u₄
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
        inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
        inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
        T : CategoryTheory.Functor C₁ C₂
        L : CategoryTheory.Functor C₁ C₃
        R : CategoryTheory.Functor C₂ C₄
        B : CategoryTheory.Functor C₃ C₄
        w : CategoryTheory.TwoSquare T L R B
        X₂ X₂' : C₂
        X₃ : C₃
        g : Quiver.Hom (R.obj X₂) (B.obj X₃)
        g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
        γ : Quiver.Hom X₂' X₂
        hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
        A A' : w.CostructuredArrowDownwards g
        φ : Quiver.Hom A A'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun A => CategoryTheory.TwoSquare.C …
      -/
      dsimp
      /-
        C₁ : Type u₁
        C₂ : Type u₂
        C₃ : Type u₃
        C₄ : Type u₄
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
        inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
        inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
        inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
        T : CategoryTheory.Functor C₁ C₂
        L : CategoryTheory.Functor C₁ C₃
        R : CategoryTheory.Functor C₂ C₄
        B : CategoryTheory.Functor C₃ C₄
        w : CategoryTheory.TwoSquare T L R B
        X₂ X₂' : C₂
        X₃ : C₃
        g : Quiver.Hom (R.obj X₂) (B.obj X₃)
        g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
        γ : Quiver.Hom X₂' X₂
        hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
        A A' : w.CostructuredArrowDownwards g
        φ : Quiver.Hom A A'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp γ …
      -/
      rw [assoc, StructuredArrow.w])) (by
      /-
        🎉 no goals
      -/
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      X₂ X₂' : C₂
      X₃ : C₃
      g : Quiver.Hom (R.obj X₂) (B.obj X₃)
      g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
      γ : Quiver.Hom X₂' X₂
      hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
      A A' : w.CostructuredArrowDownwards g
      φ : Quiver.Hom A A'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((w.structuredArrowDownwards X₂').map …
    -/
    ext
    /-
      case h
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      X₂ X₂' : C₂
      X₃ : C₃
      g : Quiver.Hom (R.obj X₂) (B.obj X₃)
      g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
      γ : Quiver.Hom X₂' X₂
      hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
      A A' : w.CostructuredArrowDownwards g
      φ : Quiver.Hom A A'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((w.structuredArrowDownwards X₂').map …
    -/
    dsimp
    /-
      case h
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      X₂ X₂' : C₂
      X₃ : C₃
      g : Quiver.Hom (R.obj X₂) (B.obj X₃)
      g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
      γ : Quiver.Hom X₂' X₂
      hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
      A A' : w.CostructuredArrowDownwards g
      φ : Quiver.Hom A A'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map φ.left.right) A'.hom.right) A. …
    -/
    rw [← CostructuredArrow.w φ, structuredArrowDownwards_map]
    /-
      case h
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      X₂ X₂' : C₂
      X₃ : C₃
      g : Quiver.Hom (R.obj X₂) (B.obj X₃)
      g' : Quiver.Hom (R.obj X₂') (B.obj X₃)
      γ : Quiver.Hom X₂' X₂
      hγ : Eq (CategoryTheory.CategoryStruct.comp (R.map γ) g) g'
      A A' : w.CostructuredArrowDownwards g
      φ : Quiver.Hom A A'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.map φ.left.right) A'.hom.right) (C …
    -/
    rfl)
    /-
      🎉 no goals
    -/
  map_id _ := rfl
  map_comp _ _ := rfl


/-- Condition on `w : TwoSquare T L R B` expressing that it is a Guitart exact square.
It is equivalent to saying that for any `X₃ : C₃`, the induced functor
`CostructuredArrow L X₃ ⥤ CostructuredArrow R (B.obj X₃)` is final (see `guitartExact_iff_final`)
or equivalently that for any `X₂ : C₂`, the induced functor
`StructuredArrow X₂ T ⥤ StructuredArrow (R.obj X₂) B` is initial (see `guitartExact_iff_initial`).
See also  `guitartExact_iff_isConnected_rightwards`, `guitartExact_iff_isConnected_downwards`
for characterizations in terms of the connectedness of auxiliary categories. -/
class GuitartExact : Prop where
  isConnected_rightwards {X₂ : C₂} {X₃ : C₃} (g : R.obj X₂ ⟶ B.obj X₃) :
    IsConnected (w.StructuredArrowRightwards g)


lemma guitartExact_iff_isConnected_rightwards :
    w.GuitartExact ↔ ∀ {X₂ : C₂} {X₃ : C₃} (g : R.obj X₂ ⟶ B.obj X₃),
      IsConnected (w.StructuredArrowRightwards g) :=
  ⟨fun h => h.isConnected_rightwards, fun h => ⟨h⟩⟩


lemma guitartExact_iff_isConnected_downwards :
    w.GuitartExact ↔ ∀ {X₂ : C₂} {X₃ : C₃} (g : R.obj X₂ ⟶ B.obj X₃),
      IsConnected (w.CostructuredArrowDownwards g) := by
  simp only [guitartExact_iff_isConnected_rightwards,
    isConnected_rightwards_iff_downwards]


instance [hw : w.GuitartExact] {X₃ : C₃} (g : CostructuredArrow R (B.obj X₃)) :
    IsConnected (StructuredArrow g (w.costructuredArrowRightwards X₃)) := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : w.GuitartExact
    X₃ : C₃
    g : CategoryTheory.CostructuredArrow R (B.obj X₃)
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow g (w.costructured …
  -/
  rw [guitartExact_iff_isConnected_rightwards] at hw
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : ∀ {X₂ : C₂} {X₃ : C₃} (g : Quiver.Hom (R.obj X₂) (B.obj X₃)), CategoryThe …
    X₃ : C₃
    g : CategoryTheory.CostructuredArrow R (B.obj X₃)
    ⊢ CategoryTheory.IsConnected (CategoryTheory.StructuredArrow g (w.costructured …
  -/
  apply hw
  /-
    🎉 no goals
  -/


instance [hw : w.GuitartExact] {X₂ : C₂} (g : StructuredArrow (R.obj X₂) B) :
    IsConnected (CostructuredArrow (w.structuredArrowDownwards X₂) g) := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : w.GuitartExact
    X₂ : C₂
    g : CategoryTheory.StructuredArrow (R.obj X₂) B
    ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (w.structuredAr …
  -/
  rw [guitartExact_iff_isConnected_downwards] at hw
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : ∀ {X₂ : C₂} {X₃ : C₃} (g : Quiver.Hom (R.obj X₂) (B.obj X₃)), CategoryThe …
    X₂ : C₂
    g : CategoryTheory.StructuredArrow (R.obj X₂) B
    ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (w.structuredAr …
  -/
  apply hw
  /-
    🎉 no goals
  -/


lemma guitartExact_iff_final :
    w.GuitartExact ↔ ∀ (X₃ : C₃), (w.costructuredArrowRightwards X₃).Final :=
  ⟨fun _ _ => ⟨fun _ => inferInstance⟩, fun _ => ⟨fun _ => inferInstance⟩⟩


instance [hw : w.GuitartExact] (X₃ : C₃) :
    (w.costructuredArrowRightwards X₃).Final := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : w.GuitartExact
    X₃ : C₃
    ⊢ (w.costructuredArrowRightwards X₃).Final
  -/
  rw [guitartExact_iff_final] at hw
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : ∀ (X₃ : C₃), (w.costructuredArrowRightwards X₃).Final
    X₃ : C₃
    ⊢ (w.costructuredArrowRightwards X₃).Final
  -/
  apply hw
  /-
    🎉 no goals
  -/


lemma guitartExact_iff_initial :
    w.GuitartExact ↔ ∀ (X₂ : C₂), (w.structuredArrowDownwards X₂).Initial :=
  ⟨fun _ _ => ⟨fun _ => inferInstance⟩, by
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      ⊢ (∀ (X₂ : C₂), (w.structuredArrowDownwards X₂).Initial) → w.GuitartExact
    -/
    rw [guitartExact_iff_isConnected_downwards]
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      ⊢ (∀ (X₂ : C₂), (w.structuredArrowDownwards X₂).Initial) → ∀ {X₂ : C₂} {X₃ : C …
    -/
    intros
    /-
      C₁ : Type u₁
      C₂ : Type u₂
      C₃ : Type u₃
      C₄ : Type u₄
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
      inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
      inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
      T : CategoryTheory.Functor C₁ C₂
      L : CategoryTheory.Functor C₁ C₃
      R : CategoryTheory.Functor C₂ C₄
      B : CategoryTheory.Functor C₃ C₄
      w : CategoryTheory.TwoSquare T L R B
      a✝ : ∀ (X₂ : C₂), (w.structuredArrowDownwards X₂).Initial
      X₂✝ : C₂
      X₃✝ : C₃
      g✝ : Quiver.Hom (R.obj X₂✝) (B.obj X₃✝)
      ⊢ CategoryTheory.IsConnected (w.CostructuredArrowDownwards g✝)
    -/
    infer_instance⟩
    /-
      🎉 no goals
    -/


instance [hw : w.GuitartExact] (X₂ : C₂) :
    (w.structuredArrowDownwards X₂).Initial := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : w.GuitartExact
    X₂ : C₂
    ⊢ (w.structuredArrowDownwards X₂).Initial
  -/
  rw [guitartExact_iff_initial] at hw
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    hw : ∀ (X₂ : C₂), (w.structuredArrowDownwards X₂).Initial
    X₂ : C₂
    ⊢ (w.structuredArrowDownwards X₂).Initial
  -/
  apply hw
  /-
    🎉 no goals
  -/


/-- When the left and right functors of a 2-square are equivalences, and the natural
transformation of the 2-square is an isomorphism, then the 2-square is Guitart exact. -/
instance (priority := 100) guitartExact_of_isEquivalence_of_isIso
    [L.IsEquivalence] [R.IsEquivalence] [IsIso w] : GuitartExact w := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    inst✝² : L.IsEquivalence
    inst✝¹ : R.IsEquivalence
    inst✝ : CategoryTheory.IsIso w
    ⊢ w.GuitartExact
  -/
  rw [guitartExact_iff_initial]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    inst✝² : L.IsEquivalence
    inst✝¹ : R.IsEquivalence
    inst✝ : CategoryTheory.IsIso w
    ⊢ ∀ (X₂ : C₂), (w.structuredArrowDownwards X₂).Initial
  -/
  intro X₂
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    inst✝² : L.IsEquivalence
    inst✝¹ : R.IsEquivalence
    inst✝ : CategoryTheory.IsIso w
    X₂ : C₂
    ⊢ (w.structuredArrowDownwards X₂).Initial
  -/
  have := StructuredArrow.isEquivalence_post X₂ T R
  have : (Comma.mapRight _ w : StructuredArrow (R.obj X₂) _ ⥤ _).IsEquivalence :=
    (Comma.mapRightIso _ (asIso w)).isEquivalence_functor
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    inst✝² : L.IsEquivalence
    inst✝¹ : R.IsEquivalence
    inst✝ : CategoryTheory.IsIso w
    X₂ : C₂
    this✝ : (CategoryTheory.StructuredArrow.post X₂ T R).IsEquivalence
    this : (CategoryTheory.Comma.mapRight (CategoryTheory.Functor.fromPUnit (R.obj …
    ⊢ (w.structuredArrowDownwards X₂).Initial
  -/
  have := StructuredArrow.isEquivalence_pre (R.obj X₂) L B
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    inst✝² : L.IsEquivalence
    inst✝¹ : R.IsEquivalence
    inst✝ : CategoryTheory.IsIso w
    X₂ : C₂
    this✝¹ : (CategoryTheory.StructuredArrow.post X₂ T R).IsEquivalence
    this✝ : (CategoryTheory.Comma.mapRight (CategoryTheory.Functor.fromPUnit (R.ob …
    this : (CategoryTheory.StructuredArrow.pre (R.obj X₂) L B).IsEquivalence
    ⊢ (w.structuredArrowDownwards X₂).Initial
  -/
  dsimp only [structuredArrowDownwards]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝³ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    inst✝² : L.IsEquivalence
    inst✝¹ : R.IsEquivalence
    inst✝ : CategoryTheory.IsIso w
    X₂ : C₂
    this✝¹ : (CategoryTheory.StructuredArrow.post X₂ T R).IsEquivalence
    this✝ : (CategoryTheory.Comma.mapRight (CategoryTheory.Functor.fromPUnit (R.ob …
    this : (CategoryTheory.StructuredArrow.pre (R.obj X₂) L B).IsEquivalence
    ⊢ ((CategoryTheory.StructuredArrow.post X₂ T R).comp ((CategoryTheory.Comma.ma …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance guitartExact_id (F : C₁ ⥤ C₂) :
    GuitartExact (TwoSquare.mk (𝟭 C₁) F F (𝟭 C₂) (𝟙 F)) := by
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    ⊢ (CategoryTheory.TwoSquare.mk (CategoryTheory.Functor.id C₁) F F (CategoryThe …
  -/
  rw [guitartExact_iff_isConnected_rightwards]
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    ⊢ ∀ {X₂ : C₁} {X₃ : C₂} (g : Quiver.Hom (F.obj X₂) ((CategoryTheory.Functor.id …
  -/
  intro X₂ X₃ (g : F.obj X₂ ⟶ X₃)
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    X₂ : C₁
    X₃ : C₂
    g : Quiver.Hom (F.obj X₂) X₃
    ⊢ CategoryTheory.IsConnected ((CategoryTheory.TwoSquare.mk (CategoryTheory.Fun …
  -/
  let Z := StructuredArrowRightwards (TwoSquare.mk (𝟭 C₁) F F (𝟭 C₂) (𝟙 F)) g
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    X₂ : C₁
    X₃ : C₂
    g : Quiver.Hom (F.obj X₂) X₃
    Z : Type (max (max u₁ v₂) v₁) := (CategoryTheory.TwoSquare.mk (CategoryTheory. …
    ⊢ CategoryTheory.IsConnected ((CategoryTheory.TwoSquare.mk (CategoryTheory.Fun …
  -/
  let X₀ : Z := StructuredArrow.mk (Y := CostructuredArrow.mk g) (CostructuredArrow.homMk (𝟙 _))
  have φ : ∀ (X : Z), X₀ ⟶ X := fun X =>
    StructuredArrow.homMk (CostructuredArrow.homMk X.hom.left
      (by simpa using CostructuredArrow.w X.hom))
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    X₂ : C₁
    X₃ : C₂
    g : Quiver.Hom (F.obj X₂) X₃
    Z : Type (max (max u₁ v₂) v₁) := (CategoryTheory.TwoSquare.mk (CategoryTheory. …
    X₀ : Z := CategoryTheory.StructuredArrow.mk (CategoryTheory.CostructuredArrow. …
    φ : (X : Z) → Quiver.Hom X₀ X
    ⊢ CategoryTheory.IsConnected ((CategoryTheory.TwoSquare.mk (CategoryTheory.Fun …
  -/
  have : Nonempty Z := ⟨X₀⟩
  /-
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    X₂ : C₁
    X₃ : C₂
    g : Quiver.Hom (F.obj X₂) X₃
    Z : Type (max (max u₁ v₂) v₁) := (CategoryTheory.TwoSquare.mk (CategoryTheory. …
    X₀ : Z := CategoryTheory.StructuredArrow.mk (CategoryTheory.CostructuredArrow. …
    φ : (X : Z) → Quiver.Hom X₀ X
    this : Nonempty Z
    ⊢ CategoryTheory.IsConnected ((CategoryTheory.TwoSquare.mk (CategoryTheory.Fun …
  -/
  apply zigzag_isConnected
  /-
    case h
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    X₂ : C₁
    X₃ : C₂
    g : Quiver.Hom (F.obj X₂) X₃
    Z : Type (max (max u₁ v₂) v₁) := (CategoryTheory.TwoSquare.mk (CategoryTheory. …
    X₀ : Z := CategoryTheory.StructuredArrow.mk (CategoryTheory.CostructuredArrow. …
    φ : (X : Z) → Quiver.Hom X₀ X
    this : Nonempty Z
    ⊢ ∀ (j₁ j₂ : (CategoryTheory.TwoSquare.mk (CategoryTheory.Functor.id C₁) F F ( …
  -/
  intro X Y
  /-
    case h
    C₁ : Type u₁
    C₂ : Type u₂
    C₃ : Type u₃
    C₄ : Type u₄
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C₁
    inst✝² : CategoryTheory.Category.{v₂, u₂} C₂
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C₃
    inst✝ : CategoryTheory.Category.{v₄, u₄} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    w : CategoryTheory.TwoSquare T L R B
    F : CategoryTheory.Functor C₁ C₂
    X₂ : C₁
    X₃ : C₂
    g : Quiver.Hom (F.obj X₂) X₃
    Z : Type (max (max u₁ v₂) v₁) := (CategoryTheory.TwoSquare.mk (CategoryTheory. …
    X₀ : Z := CategoryTheory.StructuredArrow.mk (CategoryTheory.CostructuredArrow. …
    φ : (X : Z) → Quiver.Hom X₀ X
    this : Nonempty Z
    X Y : (CategoryTheory.TwoSquare.mk (CategoryTheory.Functor.id C₁) F F (Categor …
    ⊢ CategoryTheory.Zigzag X Y
  -/
  exact Zigzag.of_inv_hom (φ X) (φ Y)
  /-
    🎉 no goals
  -/


