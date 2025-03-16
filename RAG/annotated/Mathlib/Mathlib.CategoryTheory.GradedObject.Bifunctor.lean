/-- Given a bifunctor `F : C₁ ⥤ C₂ ⥤ C₃` and types `I` and `J`, this is the obvious
functor `GradedObject I C₁ ⥤ GradedObject J C₂ ⥤ GradedObject (I × J) C₃`. -/
@[simps]
def mapBifunctor (I J : Type*) :
    GradedObject I C₁ ⥤ GradedObject J C₂ ⥤ GradedObject (I × J) C₃ where
  obj X :=
    { obj := fun Y ij => (F.obj (X ij.1)).obj (Y ij.2)
      map := fun φ ij => (F.obj (X ij.1)).map (φ ij.2) }
  map φ :=
    { app := fun Y ij => (F.map (φ ij.1)).app (Y ij.2) }


/-- Given a bifunctor `F : C₁ ⥤ C₂ ⥤ C₃`, graded objects `X : GradedObject I C₁` and
 `Y : GradedObject J C₂` and a map `p : I × J → K`, this is the `K`-graded object sending
`k` to the coproduct of `(F.obj (X i)).obj (Y j)` for `p ⟨i, j⟩ = k`. -/
noncomputable def mapBifunctorMapObj (X : GradedObject I C₁) (Y : GradedObject J C₂)
  [HasMap (((mapBifunctor F I J).obj X).obj Y) p] : GradedObject K C₃ :=
    (((mapBifunctor F I J).obj X).obj Y).mapObj p


/-- The inclusion of `(F.obj (X i)).obj (Y j)` in `mapBifunctorMapObj F p X Y k`
when `i + j = k`. -/
noncomputable def ιMapBifunctorMapObj
    (X : GradedObject I C₁) (Y : GradedObject J C₂)
    [HasMap (((mapBifunctor F I J).obj X).obj Y) p]
    (i : I) (j : J) (k : K) (h : p ⟨i, j⟩ = k) :
    (F.obj (X i)).obj (Y j) ⟶ mapBifunctorMapObj F p X Y k :=
  (((mapBifunctor F I J).obj X).obj Y).ιMapObj p ⟨i, j⟩ k h


/-- The maps `mapBifunctorMapObj F p X₁ Y₁ ⟶ mapBifunctorMapObj F p X₂ Y₂` which express
the functoriality of `mapBifunctorMapObj`, see `mapBifunctorMap`. -/
noncomputable def mapBifunctorMapMap {X₁ X₂ : GradedObject I C₁} (f : X₁ ⟶ X₂)
    {Y₁ Y₂ : GradedObject J C₂} (g : Y₁ ⟶ Y₂)
    [HasMap (((mapBifunctor F I J).obj X₁).obj Y₁) p]
    [HasMap (((mapBifunctor F I J).obj X₂).obj Y₂) p] :
    mapBifunctorMapObj F p X₁ Y₁ ⟶ mapBifunctorMapObj F p X₂ Y₂ :=
  GradedObject.mapMap (((mapBifunctor F I J).map f).app Y₁ ≫
    ((mapBifunctor F I J).obj X₂).map g) p


@[reassoc (attr := simp)]
lemma ι_mapBifunctorMapMap {X₁ X₂ : GradedObject I C₁} (f : X₁ ⟶ X₂)
    {Y₁ Y₂ : GradedObject J C₂} (g : Y₁ ⟶ Y₂)
    [HasMap (((mapBifunctor F I J).obj X₁).obj Y₁) p]
    [HasMap (((mapBifunctor F I J).obj X₂).obj Y₂) p]
    (i : I) (j : J) (k : K) (h : p ⟨i, j⟩ = k) :
    ιMapBifunctorMapObj F p X₁ Y₁ i j k h ≫ mapBifunctorMapMap F p f g k =
      (F.map (f i)).app (Y₁ j) ≫ (F.obj (X₂ i)).map (g j) ≫
        ιMapBifunctorMapObj F p X₂ Y₂ i j k h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_9, u_3} C₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
    I : Type u_4
    J : Type u_5
    K : Type u_6
    p : Prod I J → K
    X₁ X₂ : CategoryTheory.GradedObject I C₁
    f : Quiver.Hom X₁ X₂
    Y₁ Y₂ : CategoryTheory.GradedObject J C₂
    g : Quiver.Hom Y₁ Y₂
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X₁).obj Y₁).Ha …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X₂).obj Y₂).Has …
    i : I
    j : J
    k : K
    h : Eq (p { fst := i, snd := j }) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  simp [ιMapBifunctorMapObj, mapBifunctorMapMap]
  /-
    🎉 no goals
  -/


@[ext]
lemma mapBifunctorMapObj_ext {X : GradedObject I C₁} {Y : GradedObject J C₂} {A : C₃} {k : K}
    [HasMap (((mapBifunctor F I J).obj X).obj Y) p]
    {f g : mapBifunctorMapObj F p X Y k ⟶ A}
    (h : ∀ (i : I) (j : J) (hij : p ⟨i, j⟩ = k),
      ιMapBifunctorMapObj F p X Y i j k hij ≫ f = ιMapBifunctorMapObj F p X Y i j k hij ≫ g) :
    f = g := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝³ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} C₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
    I : Type u_4
    J : Type u_5
    K : Type u_6
    p : Prod I J → K
    X : CategoryTheory.GradedObject I C₁
    Y : CategoryTheory.GradedObject J C₂
    A : C₃
    k : K
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X).obj Y).HasMa …
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj F p X Y k) A
    h : ∀ (i : I) (j : J) (hij : Eq (p { fst := i, snd := j }) k), Eq (CategoryThe …
    ⊢ Eq f g
  -/
  apply mapObj_ext
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝³ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} C₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
    I : Type u_4
    J : Type u_5
    K : Type u_6
    p : Prod I J → K
    X : CategoryTheory.GradedObject I C₁
    Y : CategoryTheory.GradedObject J C₂
    A : C₃
    k : K
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X).obj Y).HasMa …
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj F p X Y k) A
    h : ∀ (i : I) (j : J) (hij : Eq (p { fst := i, snd := j }) k), Eq (CategoryThe …
    ⊢ ∀ (i : Prod I J) (hij : Eq (p i) k), Eq (CategoryTheory.CategoryStruct.comp  …
  -/
  rintro ⟨i, j⟩ hij
  /-
    case hfg.mk
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝³ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} C₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
    I : Type u_4
    J : Type u_5
    K : Type u_6
    p : Prod I J → K
    X : CategoryTheory.GradedObject I C₁
    Y : CategoryTheory.GradedObject J C₂
    A : C₃
    k : K
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X).obj Y).HasMa …
    f g : Quiver.Hom (CategoryTheory.GradedObject.mapBifunctorMapObj F p X Y k) A
    h : ∀ (i : I) (j : J) (hij : Eq (p { fst := i, snd := j }) k), Eq (CategoryThe …
    i : I
    j : J
    hij : Eq (p { fst := i, snd := j }) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.GradedObject.mapBi …
  -/
  exact h i j hij
  /-
    🎉 no goals
  -/


variable {F p} in
/-- Constructor for morphisms from `mapBifunctorMapObj F p X Y k`. -/
noncomputable def mapBifunctorMapObjDesc
    {X : GradedObject I C₁} {Y : GradedObject J C₂} {A : C₃} {k : K}
    [HasMap (((mapBifunctor F I J).obj X).obj Y) p]
    (f : ∀ (i : I) (j : J) (_ : p ⟨i, j⟩ = k), (F.obj (X i)).obj (Y j) ⟶ A) :
    mapBifunctorMapObj F p X Y k ⟶ A :=
  descMapObj _ _ (fun ⟨i, j⟩ hij => f i j hij)


@[reassoc (attr := simp)]
lemma ι_mapBifunctorMapObjDesc {X : GradedObject I C₁} {Y : GradedObject J C₂} {A : C₃} {k : K}
    [HasMap (((mapBifunctor F I J).obj X).obj Y) p]
    (f : ∀ (i : I) (j : J) (_ : p ⟨i, j⟩ = k), (F.obj (X i)).obj (Y j) ⟶ A)
    (i : I) (j : J) (hij : p ⟨i, j⟩ = k) :
    ιMapBifunctorMapObj F p X Y i j k hij ≫ mapBifunctorMapObjDesc f = f i j hij := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    inst✝³ : CategoryTheory.Category.{u_9, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_7, u_3} C₃
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
    I : Type u_4
    J : Type u_5
    K : Type u_6
    p : Prod I J → K
    X : CategoryTheory.GradedObject I C₁
    Y : CategoryTheory.GradedObject J C₂
    A : C₃
    k : K
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X).obj Y).HasMa …
    f : (i : I) → (j : J) → Eq (p { fst := i, snd := j }) k → Quiver.Hom ((F.obj ( …
    i : I
    j : J
    hij : Eq (p { fst := i, snd := j }) k
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  apply ι_descMapObj
  /-
    🎉 no goals
  -/


/-- The isomorphism `mapBifunctorMapObj F p X₁ Y₁ ≅ mapBifunctorMapObj F p X₂ Y₂`
induced by isomorphisms `X₁ ≅ X₂` and `Y₁ ≅ Y₂`. -/
@[simps]
noncomputable def mapBifunctorMapMapIso (e : X₁ ≅ X₂) (e' : Y₁ ≅ Y₂) :
    mapBifunctorMapObj F p X₁ Y₁ ≅ mapBifunctorMapObj F p X₂ Y₂ where
  hom := mapBifunctorMapMap F p e.hom e'.hom
  inv := mapBifunctorMapMap F p e.inv e'.inv
                   /-
                     C₁ : Type u_1
                     C₂ : Type u_2
                     C₃ : Type u_3
                     inst✝⁴ : CategoryTheory.Category.{?u.38460, u_1} C₁
                     inst✝³ : CategoryTheory.Category.{?u.38464, u_2} C₂
                     inst✝² : CategoryTheory.Category.{?u.38468, u_3} C₃
                     F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
                     I : Type u_4
                     J : Type u_5
                     K : Type u_6
                     p : Prod I J → K
                     X₁ X₂ : CategoryTheory.GradedObject I C₁
                     Y₁ Y₂ : CategoryTheory.GradedObject J C₂
                     inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X₁).obj Y₁).Ha …
                     inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X₂).obj Y₂).Has …
                     e : CategoryTheory.Iso X₁ X₂
                     e' : CategoryTheory.Iso Y₁ Y₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapBifun …
                   -/
  hom_inv_id := by ext; simp
                        /-
                          🎉 no goals
                        -/
                   /-
                     C₁ : Type u_1
                     C₂ : Type u_2
                     C₃ : Type u_3
                     inst✝⁴ : CategoryTheory.Category.{?u.38460, u_1} C₁
                     inst✝³ : CategoryTheory.Category.{?u.38464, u_2} C₂
                     inst✝² : CategoryTheory.Category.{?u.38468, u_3} C₃
                     F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
                     I : Type u_4
                     J : Type u_5
                     K : Type u_6
                     p : Prod I J → K
                     X₁ X₂ : CategoryTheory.GradedObject I C₁
                     Y₁ Y₂ : CategoryTheory.GradedObject J C₂
                     inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X₁).obj Y₁).Ha …
                     inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I J).obj X₂).obj Y₂).Has …
                     e : CategoryTheory.Iso X₁ X₂
                     e' : CategoryTheory.Iso Y₁ Y₂
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.mapBifun …
                   -/
  inv_hom_id := by ext; simp
                        /-
                          🎉 no goals
                        -/


instance (f : X₁ ⟶ X₂) (g : Y₁ ⟶ Y₂) [IsIso f] [IsIso g] :
    IsIso (mapBifunctorMapMap F p f g) :=
  (inferInstance : IsIso (mapBifunctorMapMapIso F p (asIso f) (asIso g)).hom)


/-- Given a bifunctor `F : C₁ ⥤ C₂ ⥤ C₃` and a map `p : I × J → K`, this is the
functor `GradedObject I C₁ ⥤ GradedObject J C₂ ⥤ GradedObject K C₃` sending
`X : GradedObject I C₁` and `Y : GradedObject J C₂` to the `K`-graded object sending
`k` to the coproduct of `(F.obj (X i)).obj (Y j)` for `p ⟨i, j⟩ = k`. -/
@[simps]
noncomputable def mapBifunctorMap [∀ X Y, HasMap (((mapBifunctor F I J).obj X).obj Y) p] :
    GradedObject I C₁ ⥤ GradedObject J C₂ ⥤ GradedObject K C₃ where
  obj X :=
    { obj := fun Y => mapBifunctorMapObj F p X Y
      map := fun ψ => mapBifunctorMapMap F p (𝟙 X) ψ }
  map {X₁ X₂} φ :=
    { app := fun Y => mapBifunctorMapMap F p φ (𝟙 Y)
      naturality := fun {Y₁ Y₂} ψ => by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₃ : Type u_3
          inst✝³ : CategoryTheory.Category.{?u.46481, u_1} C₁
          inst✝² : CategoryTheory.Category.{?u.46485, u_2} C₂
          inst✝¹ : CategoryTheory.Category.{?u.46489, u_3} C₃
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₃)
          I : Type u_4
          J : Type u_5
          K : Type u_6
          p : Prod I J → K
          inst✝ : ∀ (X : CategoryTheory.GradedObject I C₁) (Y : CategoryTheory.GradedObj …
          X₁ X₂ : CategoryTheory.GradedObject I C₁
          φ : Quiver.Hom X₁ X₂
          Y₁ Y₂ : CategoryTheory.GradedObject J C₂
          ψ : Quiver.Hom Y₁ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X => { obj := fun Y => Categor …
        -/
        dsimp
        simp only [Functor.map_id, NatTrans.id_app, id_comp, comp_id,
          ← mapMap_comp, NatTrans.naturality] }


