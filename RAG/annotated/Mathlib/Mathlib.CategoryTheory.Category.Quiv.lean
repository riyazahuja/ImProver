/-- Category of quivers. -/
@[nolint checkUnivs]
def Quiv :=
  Bundled Quiver.{v + 1, u}


instance : CoeSort Quiv (Type u) where coe := Bundled.α


instance str' (C : Quiv.{v, u}) : Quiver.{v + 1, u} C :=
  C.str


/-- Construct a bundled `Quiv` from the underlying type and the typeclass. -/
def of (C : Type u) [Quiver.{v + 1} C] : Quiv.{v, u} :=
  Bundled.of C


instance : Inhabited Quiv :=
  ⟨Quiv.of (Quiver.Empty PEmpty)⟩


/-- Category structure on `Quiv` -/
instance category : LargeCategory.{max v u} Quiv.{v, u} where
  Hom C D := Prefunctor C D
  id C := Prefunctor.id C
  comp F G := Prefunctor.comp F G


/-- The forgetful functor from categories to quivers. -/
@[simps]
def forget : Cat.{v, u} ⥤ Quiv.{v, u} where
  obj C := Quiv.of C
  map F := F.toPrefunctor


/-- The identity in the category of quivers equals the identity prefunctor.-/
theorem id_eq_id (X : Quiv) : 𝟙 X = 𝟭q X := rfl


/-- Composition in the category of quivers equals prefunctor composition.-/
theorem comp_eq_comp {X Y Z : Quiv} (F : X ⟶ Y) (G : Y ⟶ Z) : F ≫ G = F ⋙q G := rfl


/-- The functor sending each quiver to its path category. -/
@[simps]
def free : Quiv.{v, u} ⥤ Cat.{max u v, u} where
  obj V := Cat.of (Paths V)
  map F :=
    { obj := fun X => F.obj X
      map := fun f => F.mapPath f
      map_comp := fun f g => F.mapPath_comp f g }
  map_id V := by
    /-
      V : CategoryTheory.Quiv
      ⊢ Eq ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑V), map : …
    -/
    change (show Paths V ⥤ _ from _) = _
    /-
      V : CategoryTheory.Quiv
      ⊢ Eq (letFun ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑V …
    -/
    ext
      /-
        case h_obj.h
        V : CategoryTheory.Quiv
        x✝ : CategoryTheory.Paths ↑V
        ⊢ Eq ((letFun ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑ …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h
        V : CategoryTheory.Quiv
        a✝ b✝ : ↑V
        e✝ : Quiver.Hom a✝ b✝
        ⊢ Eq ((letFun ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑ …
      -/
    · exact eq_conj_eqToHom _
      /-
        🎉 no goals
      -/
  map_comp {U _ _} F G := by
    /-
      U x✝¹ x✝ : CategoryTheory.Quiv
      F : Quiver.Hom U x✝¹
      G : Quiver.Hom x✝¹ x✝
      ⊢ Eq ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑V), map : …
    -/
    change (show Paths U ⥤ _ from _) = _
    /-
      U x✝¹ x✝ : CategoryTheory.Quiv
      F : Quiver.Hom U x✝¹
      G : Quiver.Hom x✝¹ x✝
      ⊢ Eq (letFun ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑V …
    -/
    ext
      /-
        case h_obj.h
        U x✝² x✝¹ : CategoryTheory.Quiv
        F : Quiver.Hom U x✝²
        G : Quiver.Hom x✝² x✝¹
        x✝ : CategoryTheory.Paths ↑U
        ⊢ Eq ((letFun ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑ …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case h
        U x✝¹ x✝ : CategoryTheory.Quiv
        F : Quiver.Hom U x✝¹
        G : Quiver.Hom x✝¹ x✝
        a✝ b✝ : ↑U
        e✝ : Quiver.Hom a✝ b✝
        ⊢ Eq ((letFun ({ obj := fun V => CategoryTheory.Cat.of (CategoryTheory.Paths ↑ …
      -/
    · exact eq_conj_eqToHom _
      /-
        🎉 no goals
      -/


/-- An isomorphism of quivers defines an equivalence on carrier types. -/
@[simps]
def equivOfIso : V ≃ W where
  toFun := e.hom.obj
  invFun := e.inv.obj
  left_inv := Prefunctor.congr_obj e.hom_inv_id
  right_inv := Prefunctor.congr_obj e.inv_hom_id


@[simp]
lemma inv_obj_hom_obj_of_iso (X : V) : e.inv.obj (e.hom.obj X) = X := (equivOfIso e).left_inv X


@[simp]
lemma hom_obj_inv_obj_of_iso (Y : W) : e.hom.obj (e.inv.obj Y) = Y := (equivOfIso e).right_inv Y


lemma hom_map_inv_map_of_iso {V W : Quiv} (e : V ≅ W) {X Y : W} (f : X ⟶ Y) :
                                                   /-
                                                     V✝ W✝ : CategoryTheory.Quiv
                                                     e✝ : CategoryTheory.Iso V✝ W✝
                                                     V W : CategoryTheory.Quiv
                                                     e : CategoryTheory.Iso V W
                                                     X Y : ↑W
                                                     f : Quiver.Hom X Y
                                                     ⊢ Eq X (e.hom.obj (e.inv.obj X))
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    e.hom.map (e.inv.map f) = Quiver.homOfEq f (by simp) (by simp) := by
                                                             /-
                                                               🎉 no goals
                                                             -/
  /-
    V W : CategoryTheory.Quiv
    e : CategoryTheory.Iso V W
    X Y : ↑W
    f : Quiver.Hom X Y
    ⊢ Eq (e.hom.map (e.inv.map f)) (Quiver.homOfEq f ⋯ ⋯)
  -/
  rw [← Prefunctor.comp_map]
  /-
    V W : CategoryTheory.Quiv
    e : CategoryTheory.Iso V W
    X Y : ↑W
    f : Quiver.Hom X Y
    ⊢ Eq ((Prefunctor.comp e.inv e.hom).map f) (Quiver.homOfEq f ⋯ ⋯)
  -/
  exact (Prefunctor.congr_hom e.inv_hom_id.symm f).symm
  /-
    🎉 no goals
  -/


lemma inv_map_hom_map_of_iso {V W : Quiv} (e : V ≅ W) {X Y : V} (f : X ⟶ Y) :
                                                   /-
                                                     V✝ W✝ : CategoryTheory.Quiv
                                                     e✝ : CategoryTheory.Iso V✝ W✝
                                                     V W : CategoryTheory.Quiv
                                                     e : CategoryTheory.Iso V W
                                                     X Y : ↑V
                                                     f : Quiver.Hom X Y
                                                     ⊢ Eq X (e.inv.obj (e.hom.obj X))
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    e.inv.map (e.hom.map f) = Quiver.homOfEq f (by simp) (by simp) :=
                                                             /-
                                                               🎉 no goals
                                                             -/
  hom_map_inv_map_of_iso e.symm f


/-- An isomorphism of quivers defines an equivalence on hom types. -/
@[simps]
def homEquivOfIso {V W : Quiv} (e : V ≅ W) {X Y : V} :
    (X ⟶ Y) ≃ (e.hom.obj X ⟶ e.hom.obj Y) where
  toFun f := e.hom.map f
                                               /-
                                                 V✝ W✝ : CategoryTheory.Quiv
                                                 e✝ : CategoryTheory.Iso V✝ W✝
                                                 V W : CategoryTheory.Quiv
                                                 e : CategoryTheory.Iso V W
                                                 X Y : ↑V
                                                 g : Quiver.Hom (e.hom.obj X) (e.hom.obj Y)
                                                 ⊢ Eq (e.inv.obj (e.hom.obj X)) X
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  invFun g := Quiver.homOfEq (e.inv.map g) (by simp) (by simp)
                                                         /-
                                                           🎉 no goals
                                                         -/
                   /-
                     V✝ W✝ : CategoryTheory.Quiv
                     e✝ : CategoryTheory.Iso V✝ W✝
                     V W : CategoryTheory.Quiv
                     e : CategoryTheory.Iso V W
                     X Y : ↑V
                     f : Quiver.Hom X Y
                     ⊢ Eq ((fun g => Quiver.homOfEq (e.inv.map g) ⋯ ⋯) ((fun f => e.hom.map f) f)) f
                   -/
  left_inv f := by simp [inv_map_hom_map_of_iso]
                   /-
                     🎉 no goals
                   -/
                    /-
                      V✝ W✝ : CategoryTheory.Quiv
                      e✝ : CategoryTheory.Iso V✝ W✝
                      V W : CategoryTheory.Quiv
                      e : CategoryTheory.Iso V W
                      X Y : ↑V
                      g : Quiver.Hom (e.hom.obj X) (e.hom.obj Y)
                      ⊢ Eq ((fun f => e.hom.map f) ((fun g => Quiver.homOfEq (e.inv.map g) ⋯ ⋯) g)) g
                    -/
  right_inv g := by simp [hom_map_inv_map_of_iso]
                    /-
                      🎉 no goals
                    -/


include he in
@[simp]
lemma homOfEq_map_homOfEq {X Y : V} (f : X ⟶ Y) {X' Y' : V} (hX : X = X') (hY : Y = Y')
    {X'' Y'' : W} (hX' : e X' = X'') (hY' : e Y' = Y'') :
    Quiver.homOfEq (he _ _ (Quiver.homOfEq f hX hY)) hX' hY' =
                                    /-
                                      V W : Type u
                                      inst✝¹ : Quiver V
                                      inst✝ : Quiver W
                                      e : Equiv V W
                                      he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
                                      X Y : V
                                      f : Quiver.Hom X Y
                                      X' Y' : V
                                      hX : Eq X X'
                                      hY : Eq Y Y'
                                      X'' Y'' : W
                                      hX' : Eq (e X') X''
                                      hY' : Eq (e Y') Y''
                                      ⊢ Eq (e X) X''
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
      Quiver.homOfEq (he _ _ f) (by rw [hX, hX']) (by rw [hY, hY']) := by
                                                      /-
                                                        🎉 no goals
                                                      -/
  /-
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    e : Equiv V W
    he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
    X Y : V
    f : Quiver.Hom X Y
    X' Y' : V
    hX : Eq X X'
    hY : Eq Y Y'
    X'' Y'' : W
    hX' : Eq (e X') X''
    hY' : Eq (e Y') Y''
    ⊢ Eq (Quiver.homOfEq ((he X' Y') (Quiver.homOfEq f hX hY)) hX' hY') (Quiver.ho …
  -/
  subst hX hY hX' hY'
  /-
    V W : Type u
    inst✝¹ : Quiver V
    inst✝ : Quiver W
    e : Equiv V W
    he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
    X Y : V
    f : Quiver.Hom X Y
    ⊢ Eq (Quiver.homOfEq ((he X Y) (Quiver.homOfEq f ⋯ ⋯)) ⋯ ⋯) (Quiver.homOfEq (( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Compatible equivalences of types and hom-types induce an isomorphism of quivers. -/
def isoOfEquiv : Quiv.of V ≅ Quiv.of W where
  hom := Prefunctor.mk e (he _ _)
  inv :=
    { obj := e.symm
                                                         /-
                                                           V W : Type u
                                                           inst✝¹ : Quiver V
                                                           inst✝ : Quiver W
                                                           e : Equiv V W
                                                           he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
                                                           X Y : ↑(CategoryTheory.Quiv.of W)
                                                           f : Quiver.Hom X Y
                                                           ⊢ Eq X (e (e.symm X))
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
      map {X Y} f := (he _ _).symm (Quiver.homOfEq f (by simp) (by simp)) }
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  hom_inv_id := Prefunctor.ext' e.left_inv (fun X Y f ↦ by
    /-
      V W : Type u
      inst✝¹ : Quiver V
      inst✝ : Quiver W
      e : Equiv V W
      he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
      X Y : ↑(CategoryTheory.Quiv.of V)
      f : Quiver.Hom X Y
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { obj := ⇑e, map := fun {X Y} => ⇑(h …
    -/
    dsimp [Quiv.id_eq_id, Quiv.comp_eq_comp]
    /-
      V W : Type u
      inst✝¹ : Quiver V
      inst✝ : Quiver W
      e : Equiv V W
      he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
      X Y : ↑(CategoryTheory.Quiv.of V)
      f : Quiver.Hom X Y
      ⊢ Eq ((he (e.symm (e X)) (e.symm (e Y))).symm (Quiver.homOfEq ((he X Y) f) ⋯ ⋯ …
    -/
    apply (he _ _).injective
    /-
      case a
      V W : Type u
      inst✝¹ : Quiver V
      inst✝ : Quiver W
      e : Equiv V W
      he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
      X Y : ↑(CategoryTheory.Quiv.of V)
      f : Quiver.Hom X Y
      ⊢ Eq ((he (e.symm (e X)) (e.symm (e Y))) ((he (e.symm (e X)) (e.symm (e Y))).s …
    -/
    apply Quiver.homOfEq_injective (X' := e X) (Y' := e Y) (by simp) (by simp)
    /-
      case a
      V W : Type u
      inst✝¹ : Quiver V
      inst✝ : Quiver W
      e : Equiv V W
      he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
      X Y : ↑(CategoryTheory.Quiv.of V)
      f : Quiver.Hom X Y
      ⊢ Eq (Quiver.homOfEq ((he (e.symm (e X)) (e.symm (e Y))) ((he (e.symm (e X)) ( …
    -/
    simp)
    /-
      🎉 no goals
    -/
                                                /-
                                                  V W : Type u
                                                  inst✝¹ : Quiver V
                                                  inst✝ : Quiver W
                                                  e : Equiv V W
                                                  he : (X Y : V) → Equiv (Quiver.Hom X Y) (Quiver.Hom (e X) (e Y))
                                                  ⊢ ∀ (X Y : ↑(CategoryTheory.Quiv.of W)) (f : Quiver.Hom X Y), Eq ((CategoryThe …
                                                -/
  inv_hom_id := Prefunctor.ext' e.right_inv (by simp [Quiv.id_eq_id, Quiv.comp_eq_comp])
                                                /-
                                                  🎉 no goals
                                                -/


/-- Any prefunctor into a category lifts to a functor from the path category. -/
@[simps]
def lift {V : Type u} [Quiver.{v + 1} V] {C : Type*} [Category C] (F : Prefunctor V C) :
    Paths V ⥤ C where
  obj X := F.obj X
  map f := composePath (F.mapPath f)

-- We might construct `of_lift_iso_self : Paths.of ⋙ lift F ≅ F`
-- (and then show that `lift F` is initial amongst such functors)
-- but it would require lifting quite a bit of machinery to quivers!

/--
The adjunction between forming the free category on a quiver, and forgetting a category to a quiver.
-/
def adj : Cat.free ⊣ Quiv.forget :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun V C =>
        { toFun := fun F => Paths.of.comp F.toPrefunctor
          invFun := fun F => @lift V _ C _ F
                                                         /-
                                                           V : CategoryTheory.Quiv
                                                           C : CategoryTheory.Cat
                                                           F : Quiver.Hom (CategoryTheory.Cat.free.obj V) C
                                                           ⊢ ∀ (a b : ↑V) (e : Quiver.Hom a b), Eq (((fun F => CategoryTheory.Quiv.lift F …
                                                         -/
          left_inv := fun F => Paths.ext_functor rfl (by simp)
                                                         /-
                                                           🎉 no goals
                                                         -/
          right_inv := by
            /-
              V : CategoryTheory.Quiv
              C : CategoryTheory.Cat
              ⊢ Function.RightInverse (fun F => CategoryTheory.Quiv.lift F) fun F => Categor …
            -/
            rintro ⟨obj, map⟩
            /-
              case mk
              V : CategoryTheory.Quiv
              C : CategoryTheory.Cat
              obj : ↑V → ↑(CategoryTheory.Quiv.forget.obj C)
              map : {X Y : ↑V} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
              ⊢ Eq ((fun F => CategoryTheory.Paths.of.comp F.toPrefunctor) ((fun F => Catego …
            -/
            dsimp only [Prefunctor.comp]
            /-
              case mk
              V : CategoryTheory.Quiv
              C : CategoryTheory.Cat
              obj : ↑V → ↑(CategoryTheory.Quiv.forget.obj C)
              map : {X Y : ↑V} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
              ⊢ Eq { obj := fun X => (CategoryTheory.Quiv.lift { obj := obj, map := map }).o …
            -/
            congr
            /-
              case mk.e_map
              V : CategoryTheory.Quiv
              C : CategoryTheory.Cat
              obj : ↑V → ↑(CategoryTheory.Quiv.forget.obj C)
              map : {X Y : ↑V} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
              ⊢ Eq (fun {X Y} f => (CategoryTheory.Quiv.lift { obj := obj, map := map }).map …
            -/
            funext X Y f
            /-
              case mk.e_map.h.h.h
              V : CategoryTheory.Quiv
              C : CategoryTheory.Cat
              obj : ↑V → ↑(CategoryTheory.Quiv.forget.obj C)
              map : {X Y : ↑V} → Quiver.Hom X Y → Quiver.Hom (obj X) (obj Y)
              X Y : ↑V
              f : Quiver.Hom X Y
              ⊢ Eq ((CategoryTheory.Quiv.lift { obj := obj, map := map }).map (CategoryTheor …
            -/
            exact Category.id_comp _ }
            /-
              🎉 no goals
            -/
      homEquiv_naturality_left_symm := fun {V _ _} f g => by
        /-
          V x✝¹ : CategoryTheory.Quiv
          x✝ : CategoryTheory.Cat
          f : Quiver.Hom V x✝¹
          g : Quiver.Hom x✝¹ (CategoryTheory.Quiv.forget.obj x✝)
          ⊢ Eq (((fun V C => { toFun := fun F => CategoryTheory.Paths.of.comp F.toPrefun …
        -/
        change (show Paths V ⥤ _ from _) = _
        /-
          V x✝¹ : CategoryTheory.Quiv
          x✝ : CategoryTheory.Cat
          f : Quiver.Hom V x✝¹
          g : Quiver.Hom x✝¹ (CategoryTheory.Quiv.forget.obj x✝)
          ⊢ Eq (letFun (((fun V C => { toFun := fun F => CategoryTheory.Paths.of.comp F. …
        -/
        ext
          /-
            case h_obj.h
            V x✝² : CategoryTheory.Quiv
            x✝¹ : CategoryTheory.Cat
            f : Quiver.Hom V x✝²
            g : Quiver.Hom x✝² (CategoryTheory.Quiv.forget.obj x✝¹)
            x✝ : CategoryTheory.Paths ↑V
            ⊢ Eq ((letFun (((fun V C => { toFun := fun F => CategoryTheory.Paths.of.comp F …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case h
            V x✝¹ : CategoryTheory.Quiv
            x✝ : CategoryTheory.Cat
            f : Quiver.Hom V x✝¹
            g : Quiver.Hom x✝¹ (CategoryTheory.Quiv.forget.obj x✝)
            a✝ b✝ : ↑V
            e✝ : Quiver.Hom a✝ b✝
            ⊢ Eq ((letFun (((fun V C => { toFun := fun F => CategoryTheory.Paths.of.comp F …
          -/
        · apply eq_conj_eqToHom }
          /-
            🎉 no goals
          -/


