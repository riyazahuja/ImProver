/-- The type of objects for the diagram indexing a binary (co)product. -/
inductive WalkingPair : Type
  | left
  | right
  deriving DecidableEq, Inhabited


/-- The equivalence swapping left and right.
-/
def WalkingPair.swap : WalkingPair ≃ WalkingPair where
  toFun j := match j with
    | left => right
    | right => left
  invFun j := match j with
    | left => right
    | right => left
                   /-
                     j : CategoryTheory.Limits.WalkingPair
                     ⊢ Eq ((fun j => CategoryTheory.Limits.WalkingPair.swap.match_1 (fun j => Categ …
                   -/
  left_inv j := by cases j; repeat rfl
                            /-
                              🎉 no goals
                            -/
                    /-
                      j : CategoryTheory.Limits.WalkingPair
                      ⊢ Eq ((fun j => CategoryTheory.Limits.WalkingPair.swap.match_1 (fun j => Categ …
                    -/
  right_inv j := by cases j; repeat rfl
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem WalkingPair.swap_apply_left : WalkingPair.swap left = right :=
  rfl


@[simp]
theorem WalkingPair.swap_apply_right : WalkingPair.swap right = left :=
  rfl


@[simp]
theorem WalkingPair.swap_symm_apply_tt : WalkingPair.swap.symm left = right :=
  rfl


@[simp]
theorem WalkingPair.swap_symm_apply_ff : WalkingPair.swap.symm right = left :=
  rfl


/-- An equivalence from `WalkingPair` to `Bool`, sometimes useful when reindexing limits.
-/
def WalkingPair.equivBool : WalkingPair ≃ Bool where
  toFun j := match j with
    | left => true
    | right => false
  -- to match equiv.sum_equiv_sigma_bool
  invFun b := Bool.recOn b right left
                   /-
                     j : CategoryTheory.Limits.WalkingPair
                     ⊢ Eq ((fun b => Bool.recOn b CategoryTheory.Limits.WalkingPair.right CategoryT …
                   -/
  left_inv j := by cases j; repeat rfl
                            /-
                              🎉 no goals
                            -/
                    /-
                      b : Bool
                      ⊢ Eq ((fun j => CategoryTheory.Limits.WalkingPair.swap.match_1 (fun j => Bool) …
                    -/
  right_inv b := by cases b; repeat rfl
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem WalkingPair.equivBool_apply_left : WalkingPair.equivBool left = true :=
  rfl


@[simp]
theorem WalkingPair.equivBool_apply_right : WalkingPair.equivBool right = false :=
  rfl


@[simp]
theorem WalkingPair.equivBool_symm_apply_true : WalkingPair.equivBool.symm true = left :=
  rfl


@[simp]
theorem WalkingPair.equivBool_symm_apply_false : WalkingPair.equivBool.symm false = right :=
  rfl


/-- The function on the walking pair, sending the two points to `X` and `Y`. -/
def pairFunction (X Y : C) : WalkingPair → C := fun j => WalkingPair.casesOn j X Y


@[simp]
theorem pairFunction_left (X Y : C) : pairFunction X Y left = X :=
  rfl


@[simp]
theorem pairFunction_right (X Y : C) : pairFunction X Y right = Y :=
  rfl


/-- The diagram on the walking pair, sending the two points to `X` and `Y`. -/
def pair (X Y : C) : Discrete WalkingPair ⥤ C :=
  Discrete.functor fun j => WalkingPair.casesOn j X Y


@[simp]
theorem pair_obj_left (X Y : C) : (pair X Y).obj ⟨left⟩ = X :=
  rfl


@[simp]
theorem pair_obj_right (X Y : C) : (pair X Y).obj ⟨right⟩ = Y :=
  rfl


/-- The natural transformation between two functors out of the
 walking pair, specified by its components. -/
def mapPair : F ⟶ G where
  app j := match j with
    | ⟨left⟩ => f
    | ⟨right⟩ => g
                                        /-
                                          C : Type u
                                          inst✝ : CategoryTheory.Category.{v, u} C
                                          F G : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Wa …
                                          f : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.left }) (G.obj …
                                          g : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.right }) (G.ob …
                                          x✝² x✝¹ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                                          X Y : CategoryTheory.Limits.WalkingPair
                                          x✝ : Quiver.Hom { as := X } { as := Y }
                                          u : Eq { as := X }.as { as := Y }.as
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := u } }) ((f …
                                        -/
  naturality := fun ⟨X⟩ ⟨Y⟩ ⟨⟨u⟩⟩ => by aesop_cat
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem mapPair_left : (mapPair f g).app ⟨left⟩ = f :=
  rfl


@[simp]
theorem mapPair_right : (mapPair f g).app ⟨right⟩ = g :=
  rfl


/-- The natural isomorphism between two functors out of the walking pair, specified by its
components. -/
@[simps!]
def mapPairIso (f : F.obj ⟨left⟩ ≅ G.obj ⟨left⟩) (g : F.obj ⟨right⟩ ≅ G.obj ⟨right⟩) : F ≅ G :=
  NatIso.ofComponents (fun j ↦ match j with
    | ⟨left⟩ => f
    | ⟨right⟩ => g)
                     /-
                       C : Type u
                       inst✝ : CategoryTheory.Category.{v, u} C
                       F G : CategoryTheory.Functor (CategoryTheory.Discrete CategoryTheory.Limits.Wa …
                       f✝ : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.left }) (G.ob …
                       g✝ : Quiver.Hom (F.obj { as := CategoryTheory.Limits.WalkingPair.right }) (G.o …
                       f : CategoryTheory.Iso (F.obj { as := CategoryTheory.Limits.WalkingPair.left } …
                       g : CategoryTheory.Iso (F.obj { as := CategoryTheory.Limits.WalkingPair.right  …
                       X✝ Y✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                       x✝ : Quiver.Hom X✝ Y✝
                       u : Eq X✝.as Y✝.as
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map { down := { down := u } }) ((f …
                     -/
    (fun ⟨⟨u⟩⟩ => by aesop_cat)
                     /-
                       🎉 no goals
                     -/


/-- Every functor out of the walking pair is naturally isomorphic (actually, equal) to a `pair` -/
@[simps!]
def diagramIsoPair (F : Discrete WalkingPair ⥤ C) :
    F ≅ pair (F.obj ⟨WalkingPair.left⟩) (F.obj ⟨WalkingPair.right⟩) :=
  mapPairIso (Iso.refl _) (Iso.refl _)


/-- The natural isomorphism between `pair X Y ⋙ F` and `pair (F.obj X) (F.obj Y)`. -/
def pairComp (X Y : C) (F : C ⥤ D) : pair X Y ⋙ F ≅ pair (F.obj X) (F.obj Y) :=
  diagramIsoPair _


/-- A binary fan is just a cone on a diagram indexing a product. -/
abbrev BinaryFan (X Y : C) :=
  Cone (pair X Y)


/-- The first projection of a binary fan. -/
abbrev BinaryFan.fst {X Y : C} (s : BinaryFan X Y) :=
  s.π.app ⟨WalkingPair.left⟩


/-- The second projection of a binary fan. -/
abbrev BinaryFan.snd {X Y : C} (s : BinaryFan X Y) :=
  s.π.app ⟨WalkingPair.right⟩


@[simp]
theorem BinaryFan.π_app_left {X Y : C} (s : BinaryFan X Y) : s.π.app ⟨WalkingPair.left⟩ = s.fst :=
  rfl


@[simp]
theorem BinaryFan.π_app_right {X Y : C} (s : BinaryFan X Y) : s.π.app ⟨WalkingPair.right⟩ = s.snd :=
  rfl


/-- Constructs an isomorphism of `BinaryFan`s out of an isomorphism of the tips that commutes with
the projections. -/
def BinaryFan.ext {A B : C} {c c' : BinaryFan A B} (e : c.pt ≅ c'.pt)
    (h₁ : c.fst = e.hom ≫ c'.fst) (h₂ : c.snd = e.hom ≫ c'.snd) : c ≅ c' :=
                           /-
                             C : Type u
                             inst✝ : CategoryTheory.Category.{v, u} C
                             A B : C
                             c c' : CategoryTheory.Limits.BinaryFan A B
                             e : CategoryTheory.Iso c.pt c'.pt
                             h₁ : Eq c.fst (CategoryTheory.CategoryStruct.comp e.hom c'.fst)
                             h₂ : Eq c.snd (CategoryTheory.CategoryStruct.comp e.hom c'.snd)
                             j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                             ⊢ Eq (c.π.app j) (CategoryTheory.CategoryStruct.comp e.hom (c'.π.app j))
                           -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  Cones.ext e (fun j => by rcases j with ⟨⟨⟩⟩ <;> assumption)
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- A convenient way to show that a binary fan is a limit. -/
def BinaryFan.IsLimit.mk {X Y : C} (s : BinaryFan X Y)
    (lift : ∀ {T : C} (_ : T ⟶ X) (_ : T ⟶ Y), T ⟶ s.pt)
    (hl₁ : ∀ {T : C} (f : T ⟶ X) (g : T ⟶ Y), lift f g ≫ s.fst = f)
    (hl₂ : ∀ {T : C} (f : T ⟶ X) (g : T ⟶ Y), lift f g ≫ s.snd = g)
    (uniq :
      ∀ {T : C} (f : T ⟶ X) (g : T ⟶ Y) (m : T ⟶ s.pt) (_ : m ≫ s.fst = f) (_ : m ≫ s.snd = g),
        m = lift f g) :
    IsLimit s :=
  Limits.IsLimit.mk (fun t => lift (BinaryFan.fst t) (BinaryFan.snd t))
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        s : CategoryTheory.Limits.BinaryFan X Y
        lift : {T : C} → Quiver.Hom T X → Quiver.Hom T Y → Quiver.Hom T s.pt
        hl₁ : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y), Eq (CategoryTheory. …
        hl₂ : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y), Eq (CategoryTheory. …
        uniq : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y) (m : Quiver.Hom T s …
        ⊢ ∀ (s_1 : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)) (j : C …
      -/
      rintro t (rfl | rfl)
        /-
          case mk.left
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          s : CategoryTheory.Limits.BinaryFan X Y
          lift : {T : C} → Quiver.Hom T X → Quiver.Hom T Y → Quiver.Hom T s.pt
          hl₁ : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y), Eq (CategoryTheory. …
          hl₂ : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y), Eq (CategoryTheory. …
          uniq : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y) (m : Quiver.Hom T s …
          t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun t => lift (CategoryTheory.Limit …
        -/
      · exact hl₁ _ _
        /-
          🎉 no goals
        -/
        /-
          case mk.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          s : CategoryTheory.Limits.BinaryFan X Y
          lift : {T : C} → Quiver.Hom T X → Quiver.Hom T Y → Quiver.Hom T s.pt
          hl₁ : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y), Eq (CategoryTheory. …
          hl₂ : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y), Eq (CategoryTheory. …
          uniq : ∀ {T : C} (f : Quiver.Hom T X) (g : Quiver.Hom T Y) (m : Quiver.Hom T s …
          t : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun t => lift (CategoryTheory.Limit …
        -/
      · exact hl₂ _ _)
        /-
          🎉 no goals
        -/
    fun _ _ h => uniq _ _ _ (h ⟨WalkingPair.left⟩) (h ⟨WalkingPair.right⟩)


theorem BinaryFan.IsLimit.hom_ext {W X Y : C} {s : BinaryFan X Y} (h : IsLimit s) {f g : W ⟶ s.pt}
    (h₁ : f ≫ s.fst = g ≫ s.fst) (h₂ : f ≫ s.snd = g ≫ s.snd) : f = g :=
  h.hom_ext fun j => Discrete.recOn j fun j => WalkingPair.casesOn j h₁ h₂


/-- A binary cofan is just a cocone on a diagram indexing a coproduct. -/
abbrev BinaryCofan (X Y : C) := Cocone (pair X Y)


/-- The first inclusion of a binary cofan. -/
abbrev BinaryCofan.inl {X Y : C} (s : BinaryCofan X Y) := s.ι.app ⟨WalkingPair.left⟩


/-- The second inclusion of a binary cofan. -/
abbrev BinaryCofan.inr {X Y : C} (s : BinaryCofan X Y) := s.ι.app ⟨WalkingPair.right⟩


/-- Constructs an isomorphism of `BinaryCofan`s out of an isomorphism of the tips that commutes with
the injections. -/
def BinaryCofan.ext {A B : C} {c c' : BinaryCofan A B} (e : c.pt ≅ c'.pt)
    (h₁ : c.inl ≫ e.hom = c'.inl) (h₂ : c.inr ≫ e.hom = c'.inr) : c ≅ c' :=
                             /-
                               C : Type u
                               inst✝ : CategoryTheory.Category.{v, u} C
                               A B : C
                               c c' : CategoryTheory.Limits.BinaryCofan A B
                               e : CategoryTheory.Iso c.pt c'.pt
                               h₁ : Eq (CategoryTheory.CategoryStruct.comp c.inl e.hom) c'.inl
                               h₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr e.hom) c'.inr
                               j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) e.hom) (c'.ι.app j)
                             -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  Cocones.ext e (fun j => by rcases j with ⟨⟨⟩⟩ <;> assumption)
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem BinaryCofan.ι_app_left {X Y : C} (s : BinaryCofan X Y) :
    s.ι.app ⟨WalkingPair.left⟩ = s.inl := rfl


@[simp]
theorem BinaryCofan.ι_app_right {X Y : C} (s : BinaryCofan X Y) :
    s.ι.app ⟨WalkingPair.right⟩ = s.inr := rfl


/-- A convenient way to show that a binary cofan is a colimit. -/
def BinaryCofan.IsColimit.mk {X Y : C} (s : BinaryCofan X Y)
    (desc : ∀ {T : C} (_ : X ⟶ T) (_ : Y ⟶ T), s.pt ⟶ T)
    (hd₁ : ∀ {T : C} (f : X ⟶ T) (g : Y ⟶ T), s.inl ≫ desc f g = f)
    (hd₂ : ∀ {T : C} (f : X ⟶ T) (g : Y ⟶ T), s.inr ≫ desc f g = g)
    (uniq :
      ∀ {T : C} (f : X ⟶ T) (g : Y ⟶ T) (m : s.pt ⟶ T) (_ : s.inl ≫ m = f) (_ : s.inr ≫ m = g),
        m = desc f g) :
    IsColimit s :=
  Limits.IsColimit.mk (fun t => desc (BinaryCofan.inl t) (BinaryCofan.inr t))
    (by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y : C
        s : CategoryTheory.Limits.BinaryCofan X Y
        desc : {T : C} → Quiver.Hom X T → Quiver.Hom Y T → Quiver.Hom s.pt T
        hd₁ : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T), Eq (CategoryTheory. …
        hd₂ : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T), Eq (CategoryTheory. …
        uniq : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T) (m : Quiver.Hom s.p …
        ⊢ ∀ (s_1 : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)) (j : …
      -/
      rintro t (rfl | rfl)
        /-
          case mk.left
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          s : CategoryTheory.Limits.BinaryCofan X Y
          desc : {T : C} → Quiver.Hom X T → Quiver.Hom Y T → Quiver.Hom s.pt T
          hd₁ : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T), Eq (CategoryTheory. …
          hd₂ : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T), Eq (CategoryTheory. …
          uniq : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T) (m : Quiver.Hom s.p …
          t : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app { as := CategoryTheory.Limit …
        -/
      · exact hd₁ _ _
        /-
          🎉 no goals
        -/
        /-
          case mk.right
          C : Type u
          inst✝ : CategoryTheory.Category.{v, u} C
          X Y : C
          s : CategoryTheory.Limits.BinaryCofan X Y
          desc : {T : C} → Quiver.Hom X T → Quiver.Hom Y T → Quiver.Hom s.pt T
          hd₁ : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T), Eq (CategoryTheory. …
          hd₂ : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T), Eq (CategoryTheory. …
          uniq : ∀ {T : C} (f : Quiver.Hom X T) (g : Quiver.Hom Y T) (m : Quiver.Hom s.p …
          t : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app { as := CategoryTheory.Limit …
        -/
      · exact hd₂ _ _)
        /-
          🎉 no goals
        -/
    fun _ _ h => uniq _ _ _ (h ⟨WalkingPair.left⟩) (h ⟨WalkingPair.right⟩)


theorem BinaryCofan.IsColimit.hom_ext {W X Y : C} {s : BinaryCofan X Y} (h : IsColimit s)
    {f g : s.pt ⟶ W} (h₁ : s.inl ≫ f = s.inl ≫ g) (h₂ : s.inr ≫ f = s.inr ≫ g) : f = g :=
  h.hom_ext fun j => Discrete.recOn j fun j => WalkingPair.casesOn j h₁ h₂


/-- A binary fan with vertex `P` consists of the two projections `π₁ : P ⟶ X` and `π₂ : P ⟶ Y`. -/
@[simps pt]
def BinaryFan.mk {P : C} (π₁ : P ⟶ X) (π₂ : P ⟶ Y) : BinaryFan X Y where
  pt := P
  π := { app := fun | { as := j } => match j with | left => π₁ | right => π₂ }


/-- A binary cofan with vertex `P` consists of the two inclusions `ι₁ : X ⟶ P` and `ι₂ : Y ⟶ P`. -/
@[simps pt]
def BinaryCofan.mk {P : C} (ι₁ : X ⟶ P) (ι₂ : Y ⟶ P) : BinaryCofan X Y where
  pt := P
  ι := { app := fun | { as := j } => match j with | left => ι₁ | right => ι₂ }


@[simp]
theorem BinaryFan.mk_fst {P : C} (π₁ : P ⟶ X) (π₂ : P ⟶ Y) : (BinaryFan.mk π₁ π₂).fst = π₁ :=
  rfl


@[simp]
theorem BinaryFan.mk_snd {P : C} (π₁ : P ⟶ X) (π₂ : P ⟶ Y) : (BinaryFan.mk π₁ π₂).snd = π₂ :=
  rfl


@[simp]
theorem BinaryCofan.mk_inl {P : C} (ι₁ : X ⟶ P) (ι₂ : Y ⟶ P) : (BinaryCofan.mk ι₁ ι₂).inl = ι₁ :=
  rfl


@[simp]
theorem BinaryCofan.mk_inr {P : C} (ι₁ : X ⟶ P) (ι₂ : Y ⟶ P) : (BinaryCofan.mk ι₁ ι₂).inr = ι₂ :=
  rfl


/-- Every `BinaryFan` is isomorphic to an application of `BinaryFan.mk`. -/
def isoBinaryFanMk {X Y : C} (c : BinaryFan X Y) : c ≅ BinaryFan.mk c.fst c.snd :=
                                       /-
                                         C : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} C
                                         X✝ Y✝ X Y : C
                                         c : CategoryTheory.Limits.BinaryFan X Y
                                         j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                                         ⊢ Eq (c.π.app j) (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl  …
                                       -/
    Cones.ext (Iso.refl _) fun j => by cases' j with l; cases l; repeat simp
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Every `BinaryFan` is isomorphic to an application of `BinaryFan.mk`. -/
def isoBinaryCofanMk {X Y : C} (c : BinaryCofan X Y) : c ≅ BinaryCofan.mk c.inl c.inr :=
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           X✝ Y✝ X Y : C
                                           c : CategoryTheory.Limits.BinaryCofan X Y
                                           j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (c.ι.app j) (CategoryTheory.Iso.refl  …
                                         -/
    Cocones.ext (Iso.refl _) fun j => by cases' j with l; cases l; repeat simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- This is a more convenient formulation to show that a `BinaryFan` constructed using
`BinaryFan.mk` is a limit cone.
-/
def BinaryFan.isLimitMk {W : C} {fst : W ⟶ X} {snd : W ⟶ Y} (lift : ∀ s : BinaryFan X Y, s.pt ⟶ W)
    (fac_left : ∀ s : BinaryFan X Y, lift s ≫ fst = s.fst)
    (fac_right : ∀ s : BinaryFan X Y, lift s ≫ snd = s.snd)
    (uniq :
      ∀ (s : BinaryFan X Y) (m : s.pt ⟶ W) (_ : m ≫ fst = s.fst) (_ : m ≫ snd = s.snd),
        m = lift s) :
    IsLimit (BinaryFan.mk fst snd) :=
  { lift := lift
    fac := fun s j => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y W : C
        fst : Quiver.Hom W X
        snd : Quiver.Hom W Y
        lift : (s : CategoryTheory.Limits.BinaryFan X Y) → Quiver.Hom s.pt W
        fac_left : ∀ (s : CategoryTheory.Limits.BinaryFan X Y), Eq (CategoryTheory.Cat …
        fac_right : ∀ (s : CategoryTheory.Limits.BinaryFan X Y), Eq (CategoryTheory.Ca …
        uniq : ∀ (s : CategoryTheory.Limits.BinaryFan X Y) (m : Quiver.Hom s.pt W), Eq …
        s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) ((CategoryTheory.Limits.Bina …
      -/
      rcases j with ⟨⟨⟩⟩
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y W : C
        fst : Quiver.Hom W X
        snd : Quiver.Hom W Y
        lift : (s : CategoryTheory.Limits.BinaryFan X Y) → Quiver.Hom s.pt W
        fac_left : ∀ (s : CategoryTheory.Limits.BinaryFan X Y), Eq (CategoryTheory.Cat …
        fac_right : ∀ (s : CategoryTheory.Limits.BinaryFan X Y), Eq (CategoryTheory.Ca …
        uniq : ∀ (s : CategoryTheory.Limits.BinaryFan X Y) (m : Quiver.Hom s.pt W), Eq …
        s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair X Y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) ((CategoryTheory.Limits.Bina …
      -/
      exacts [fac_left s, fac_right s]
      /-
        🎉 no goals
      -/
    uniq := fun s m w => uniq s m (w ⟨WalkingPair.left⟩) (w ⟨WalkingPair.right⟩) }


/-- This is a more convenient formulation to show that a `BinaryCofan` constructed using
`BinaryCofan.mk` is a colimit cocone.
-/
def BinaryCofan.isColimitMk {W : C} {inl : X ⟶ W} {inr : Y ⟶ W}
    (desc : ∀ s : BinaryCofan X Y, W ⟶ s.pt)
    (fac_left : ∀ s : BinaryCofan X Y, inl ≫ desc s = s.inl)
    (fac_right : ∀ s : BinaryCofan X Y, inr ≫ desc s = s.inr)
    (uniq :
      ∀ (s : BinaryCofan X Y) (m : W ⟶ s.pt) (_ : inl ≫ m = s.inl) (_ : inr ≫ m = s.inr),
        m = desc s) :
    IsColimit (BinaryCofan.mk inl inr) :=
  { desc := desc
    fac := fun s j => by
      /-
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y W : C
        inl : Quiver.Hom X W
        inr : Quiver.Hom Y W
        desc : (s : CategoryTheory.Limits.BinaryCofan X Y) → Quiver.Hom W s.pt
        fac_left : ∀ (s : CategoryTheory.Limits.BinaryCofan X Y), Eq (CategoryTheory.C …
        fac_right : ∀ (s : CategoryTheory.Limits.BinaryCofan X Y), Eq (CategoryTheory. …
        uniq : ∀ (s : CategoryTheory.Limits.BinaryCofan X Y) (m : Quiver.Hom W s.pt),  …
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
        j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.BinaryCofan.m …
      -/
      rcases j with ⟨⟨⟩⟩
      /-
        case mk.left
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        X Y W : C
        inl : Quiver.Hom X W
        inr : Quiver.Hom Y W
        desc : (s : CategoryTheory.Limits.BinaryCofan X Y) → Quiver.Hom W s.pt
        fac_left : ∀ (s : CategoryTheory.Limits.BinaryCofan X Y), Eq (CategoryTheory.C …
        fac_right : ∀ (s : CategoryTheory.Limits.BinaryCofan X Y), Eq (CategoryTheory. …
        uniq : ∀ (s : CategoryTheory.Limits.BinaryCofan X Y) (m : Quiver.Hom W s.pt),  …
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.pair X Y)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.BinaryCofan.m …
      -/
      exacts [fac_left s, fac_right s]
      /-
        🎉 no goals
      -/
    uniq := fun s m w => uniq s m (w ⟨WalkingPair.left⟩) (w ⟨WalkingPair.right⟩) }


/-- If `s` is a limit binary fan over `X` and `Y`, then every pair of morphisms `f : W ⟶ X` and
    `g : W ⟶ Y` induces a morphism `l : W ⟶ s.pt` satisfying `l ≫ s.fst = f` and `l ≫ s.snd = g`.
    -/
@[simps]
def BinaryFan.IsLimit.lift' {W X Y : C} {s : BinaryFan X Y} (h : IsLimit s) (f : W ⟶ X)
    (g : W ⟶ Y) : { l : W ⟶ s.pt // l ≫ s.fst = f ∧ l ≫ s.snd = g } :=
  ⟨h.lift <| BinaryFan.mk f g, h.fac _ _, h.fac _ _⟩


/-- If `s` is a colimit binary cofan over `X` and `Y`,, then every pair of morphisms `f : X ⟶ W` and
    `g : Y ⟶ W` induces a morphism `l : s.pt ⟶ W` satisfying `s.inl ≫ l = f` and `s.inr ≫ l = g`.
    -/
@[simps]
def BinaryCofan.IsColimit.desc' {W X Y : C} {s : BinaryCofan X Y} (h : IsColimit s) (f : X ⟶ W)
    (g : Y ⟶ W) : { l : s.pt ⟶ W // s.inl ≫ l = f ∧ s.inr ≫ l = g } :=
  ⟨h.desc <| BinaryCofan.mk f g, h.fac _ _, h.fac _ _⟩


/-- Binary products are symmetric. -/
def BinaryFan.isLimitFlip {X Y : C} {c : BinaryFan X Y} (hc : IsLimit c) :
    IsLimit (BinaryFan.mk c.snd c.fst) :=
  BinaryFan.isLimitMk (fun s => hc.lift (BinaryFan.mk s.snd s.fst)) (fun _ => hc.fac _ _)
    (fun _ => hc.fac _ _) fun s _ e₁ e₂ =>
    BinaryFan.IsLimit.hom_ext hc
      (e₂.trans (hc.fac (BinaryFan.mk s.snd s.fst) ⟨WalkingPair.left⟩).symm)
      (e₁.trans (hc.fac (BinaryFan.mk s.snd s.fst) ⟨WalkingPair.right⟩).symm)


theorem BinaryFan.isLimit_iff_isIso_fst {X Y : C} (h : IsTerminal Y) (c : BinaryFan X Y) :
    Nonempty (IsLimit c) ↔ IsIso c.fst := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    h : CategoryTheory.Limits.IsTerminal Y
    c : CategoryTheory.Limits.BinaryFan X Y
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsLimit c)) (CategoryTheory.IsIso c.fst)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsTerminal Y
      c : CategoryTheory.Limits.BinaryFan X Y
      ⊢ Nonempty (CategoryTheory.Limits.IsLimit c) → CategoryTheory.IsIso c.fst
    -/
  · rintro ⟨H⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsTerminal Y
      c : CategoryTheory.Limits.BinaryFan X Y
      H : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.IsIso c.fst
    -/
    obtain ⟨l, hl, -⟩ := BinaryFan.IsLimit.lift' H (𝟙 X) (h.from X)
    exact
      ⟨⟨l,
          BinaryFan.IsLimit.hom_ext H (by simpa [hl, -Category.comp_id] using Category.comp_id _)
            (h.hom_ext _ _),
          hl⟩⟩
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsTerminal Y
      c : CategoryTheory.Limits.BinaryFan X Y
      ⊢ CategoryTheory.IsIso c.fst → Nonempty (CategoryTheory.Limits.IsLimit c)
    -/
  · intro
    exact
      ⟨BinaryFan.IsLimit.mk _ (fun f _ => f ≫ inv c.fst) (fun _ _ => by simp)
          (fun _ _ => h.hom_ext _ _) fun _ _ _ e _ => by simp [← e]⟩


theorem BinaryFan.isLimit_iff_isIso_snd {X Y : C} (h : IsTerminal X) (c : BinaryFan X Y) :
    Nonempty (IsLimit c) ↔ IsIso c.snd := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    h : CategoryTheory.Limits.IsTerminal X
    c : CategoryTheory.Limits.BinaryFan X Y
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsLimit c)) (CategoryTheory.IsIso c.snd)
  -/
  refine Iff.trans ?_ (BinaryFan.isLimit_iff_isIso_fst h (BinaryFan.mk c.snd c.fst))
  exact
    ⟨fun h => ⟨BinaryFan.isLimitFlip h.some⟩, fun h =>
      ⟨(BinaryFan.isLimitFlip h.some).ofIsoLimit (isoBinaryFanMk c).symm⟩⟩


/-- If `X' ≅ X`, then `X × Y` also is the product of `X'` and `Y`. -/
noncomputable def BinaryFan.isLimitCompLeftIso {X Y X' : C} (c : BinaryFan X Y) (f : X ⟶ X')
    [IsIso f] (h : IsLimit c) : IsLimit (BinaryFan.mk (c.fst ≫ f) c.snd) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ X Y X' : C
    c : CategoryTheory.Limits.BinaryFan X Y
    f : Quiver.Hom X X'
    inst✝ : CategoryTheory.IsIso f
    h : CategoryTheory.Limits.IsLimit c
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.BinaryFan.mk (CategoryT …
  -/
  fapply BinaryFan.isLimitMk
    /-
      case lift
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryFan X Y
      f : Quiver.Hom X X'
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsLimit c
      ⊢ (s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y).ob …
    -/
  · exact fun s => h.lift (BinaryFan.mk (s.fst ≫ inv f) s.snd)
    /-
      🎉 no goals
    -/
    /-
      case fac_left
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryFan X Y
      f : Quiver.Hom X X'
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y). …
    -/
  · intro s -- Porting note: simp timed out here
    simp only [Category.comp_id,BinaryFan.π_app_left,IsIso.inv_hom_id,
      BinaryFan.mk_fst,IsLimit.fac_assoc,eq_self_iff_true,Category.assoc]
    /-
      case fac_right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryFan X Y
      f : Quiver.Hom X X'
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y). …
    -/
  · intro s -- Porting note: simp timed out here
    /-
      case fac_right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryFan X Y
      f : Quiver.Hom X X'
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y).obj { …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.lift (CategoryTheory.Limits.Binary …
    -/
    simp only [BinaryFan.π_app_right,BinaryFan.mk_snd,eq_self_iff_true,IsLimit.fac]
    /-
      🎉 no goals
    -/
    /-
      case uniq
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryFan X Y
      f : Quiver.Hom X X'
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsLimit c
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y). …
    -/
  · intro s m e₁ e₂
     -- Porting note: simpa timed out here also
    /-
      case uniq
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryFan X Y
      f : Quiver.Hom X X'
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsLimit c
      s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y).obj { …
      m : Quiver.Hom s.pt (((CategoryTheory.Functor.const (CategoryTheory.Discrete C …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp m c.snd) s.snd
      ⊢ Eq m (h.lift (CategoryTheory.Limits.BinaryFan.mk (CategoryTheory.CategoryStr …
    -/
    apply BinaryFan.IsLimit.hom_ext h
    · simpa only
      [BinaryFan.π_app_left,BinaryFan.mk_fst,Category.assoc,IsLimit.fac,IsIso.eq_comp_inv]
      /-
        case uniq.h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y X' : C
        c : CategoryTheory.Limits.BinaryFan X Y
        f : Quiver.Hom X X'
        inst✝ : CategoryTheory.IsIso f
        h : CategoryTheory.Limits.IsLimit c
        s : CategoryTheory.Limits.BinaryFan X' ((CategoryTheory.Limits.pair X Y).obj { …
        m : Quiver.Hom s.pt (((CategoryTheory.Functor.const (CategoryTheory.Discrete C …
        e₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp m c.snd) s.snd
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m c.snd) (CategoryTheory.CategoryStru …
      -/
    · simpa only [BinaryFan.π_app_right,BinaryFan.mk_snd,IsLimit.fac]
      /-
        🎉 no goals
      -/


/-- If `Y' ≅ Y`, then `X x Y` also is the product of `X` and `Y'`. -/
noncomputable def BinaryFan.isLimitCompRightIso {X Y Y' : C} (c : BinaryFan X Y) (f : Y ⟶ Y')
    [IsIso f] (h : IsLimit c) : IsLimit (BinaryFan.mk c.fst (c.snd ≫ f)) :=
  BinaryFan.isLimitFlip <| BinaryFan.isLimitCompLeftIso _ f (BinaryFan.isLimitFlip h)


/-- Binary coproducts are symmetric. -/
def BinaryCofan.isColimitFlip {X Y : C} {c : BinaryCofan X Y} (hc : IsColimit c) :
    IsColimit (BinaryCofan.mk c.inr c.inl) :=
  BinaryCofan.isColimitMk (fun s => hc.desc (BinaryCofan.mk s.inr s.inl)) (fun _ => hc.fac _ _)
    (fun _ => hc.fac _ _) fun s _ e₁ e₂ =>
    BinaryCofan.IsColimit.hom_ext hc
      (e₂.trans (hc.fac (BinaryCofan.mk s.inr s.inl) ⟨WalkingPair.left⟩).symm)
      (e₁.trans (hc.fac (BinaryCofan.mk s.inr s.inl) ⟨WalkingPair.right⟩).symm)


theorem BinaryCofan.isColimit_iff_isIso_inl {X Y : C} (h : IsInitial Y) (c : BinaryCofan X Y) :
    Nonempty (IsColimit c) ↔ IsIso c.inl := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    h : CategoryTheory.Limits.IsInitial Y
    c : CategoryTheory.Limits.BinaryCofan X Y
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c)) (CategoryTheory.IsIso c.i …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      ⊢ Nonempty (CategoryTheory.Limits.IsColimit c) → CategoryTheory.IsIso c.inl
    -/
  · rintro ⟨H⟩
    /-
      case mp.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.IsIso c.inl
    -/
    obtain ⟨l, hl, -⟩ := BinaryCofan.IsColimit.desc' H (𝟙 X) (h.to X)
    /-
      case mp.intro.mk.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.Limits.IsColimit c
      l : Quiver.Hom c.pt X
      hl : Eq (CategoryTheory.CategoryStruct.comp c.inl l) (CategoryTheory.CategoryS …
      ⊢ CategoryTheory.IsIso c.inl
    -/
    refine ⟨⟨l, hl, BinaryCofan.IsColimit.hom_ext H (?_) (h.hom_ext _ _)⟩⟩
    /-
      case mp.intro.mk.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.Limits.IsColimit c
      l : Quiver.Hom c.pt X
      hl : Eq (CategoryTheory.CategoryStruct.comp c.inl l) (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
    -/
    rw [Category.comp_id]
    /-
      case mp.intro.mk.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.Limits.IsColimit c
      l : Quiver.Hom c.pt X
      hl : Eq (CategoryTheory.CategoryStruct.comp c.inl l) (CategoryTheory.CategoryS …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
    -/
    have e : (inl c ≫ l) ≫ inl c = 𝟙 X ≫ inl c := congrArg (·≫inl c) hl
    /-
      case mp.intro.mk.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      H : CategoryTheory.Limits.IsColimit c
      l : Quiver.Hom c.pt X
      hl : Eq (CategoryTheory.CategoryStruct.comp c.inl l) (CategoryTheory.CategoryS …
      e : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl (CategoryTheory.CategoryStruct. …
    -/
    rwa [Category.assoc,Category.id_comp] at e
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : C
      h : CategoryTheory.Limits.IsInitial Y
      c : CategoryTheory.Limits.BinaryCofan X Y
      ⊢ CategoryTheory.IsIso c.inl → Nonempty (CategoryTheory.Limits.IsColimit c)
    -/
  · intro
    exact
      ⟨BinaryCofan.IsColimit.mk _ (fun f _ => inv c.inl ≫ f)
          (fun _ _ => IsIso.hom_inv_id_assoc _ _) (fun _ _ => h.hom_ext _ _) fun _ _ _ e _ =>
          (IsIso.eq_inv_comp _).mpr e⟩


theorem BinaryCofan.isColimit_iff_isIso_inr {X Y : C} (h : IsInitial X) (c : BinaryCofan X Y) :
    Nonempty (IsColimit c) ↔ IsIso c.inr := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    h : CategoryTheory.Limits.IsInitial X
    c : CategoryTheory.Limits.BinaryCofan X Y
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsColimit c)) (CategoryTheory.IsIso c.i …
  -/
  refine Iff.trans ?_ (BinaryCofan.isColimit_iff_isIso_inl h (BinaryCofan.mk c.inr c.inl))
  exact
    ⟨fun h => ⟨BinaryCofan.isColimitFlip h.some⟩, fun h =>
      ⟨(BinaryCofan.isColimitFlip h.some).ofIsoColimit (isoBinaryCofanMk c).symm⟩⟩


/-- If `X' ≅ X`, then `X ⨿ Y` also is the coproduct of `X'` and `Y`. -/
noncomputable def BinaryCofan.isColimitCompLeftIso {X Y X' : C} (c : BinaryCofan X Y) (f : X' ⟶ X)
    [IsIso f] (h : IsColimit c) : IsColimit (BinaryCofan.mk (f ≫ c.inl) c.inr) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ X Y X' : C
    c : CategoryTheory.Limits.BinaryCofan X Y
    f : Quiver.Hom X' X
    inst✝ : CategoryTheory.IsIso f
    h : CategoryTheory.Limits.IsColimit c
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk (Categ …
  -/
  fapply BinaryCofan.isColimitMk
    /-
      case desc
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      f : Quiver.Hom X' X
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsColimit c
      ⊢ (s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y). …
    -/
  · exact fun s => h.desc (BinaryCofan.mk (inv f ≫ s.inl) s.inr)
    /-
      🎉 no goals
    -/
    /-
      case fac_left
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      f : Quiver.Hom X' X
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsColimit c
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y …
    -/
  · intro s
    -- Porting note: simp timed out here too
    simp only [IsColimit.fac,BinaryCofan.ι_app_left,eq_self_iff_true,
      Category.assoc,BinaryCofan.mk_inl,IsIso.hom_inv_id_assoc]
    /-
      case fac_right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      f : Quiver.Hom X' X
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsColimit c
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y …
    -/
  · intro s
    -- Porting note: simp timed out here too
    /-
      case fac_right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      f : Quiver.Hom X' X
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y).obj …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr (h.desc (CategoryTheory.Limits. …
    -/
    simp only [IsColimit.fac,BinaryCofan.ι_app_right,eq_self_iff_true,BinaryCofan.mk_inr]
    /-
      🎉 no goals
    -/
    /-
      case uniq
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      f : Quiver.Hom X' X
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsColimit c
      ⊢ ∀ (s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y …
    -/
  · intro s m e₁ e₂
    /-
      case uniq
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y X' : C
      c : CategoryTheory.Limits.BinaryCofan X Y
      f : Quiver.Hom X' X
      inst✝ : CategoryTheory.IsIso f
      h : CategoryTheory.Limits.IsColimit c
      s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y).obj …
      m : Quiver.Hom (((CategoryTheory.Functor.const (CategoryTheory.Discrete Catego …
      e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr m) s.inr
      ⊢ Eq m (h.desc (CategoryTheory.Limits.BinaryCofan.mk (CategoryTheory.CategoryS …
    -/
    apply BinaryCofan.IsColimit.hom_ext h
      /-
        case uniq.h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y X' : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        f : Quiver.Hom X' X
        inst✝ : CategoryTheory.IsIso f
        h : CategoryTheory.Limits.IsColimit c
        s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y).obj …
        m : Quiver.Hom (((CategoryTheory.Functor.const (CategoryTheory.Discrete Catego …
        e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr m) s.inr
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inl m) (CategoryTheory.CategoryStru …
      -/
    · rw [← cancel_epi f]
    -- Porting note: simp timed out here too
      simpa only [IsColimit.fac,BinaryCofan.ι_app_left,eq_self_iff_true,
      Category.assoc,BinaryCofan.mk_inl,IsIso.hom_inv_id_assoc] using e₁
    -- Porting note: simp timed out here too
      /-
        case uniq.h₂
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y X' : C
        c : CategoryTheory.Limits.BinaryCofan X Y
        f : Quiver.Hom X' X
        inst✝ : CategoryTheory.IsIso f
        h : CategoryTheory.Limits.IsColimit c
        s : CategoryTheory.Limits.BinaryCofan X' ((CategoryTheory.Limits.pair X Y).obj …
        m : Quiver.Hom (((CategoryTheory.Functor.const (CategoryTheory.Discrete Catego …
        e₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp c.inr m) s.inr
        ⊢ Eq (CategoryTheory.CategoryStruct.comp c.inr m) (CategoryTheory.CategoryStru …
      -/
    · simpa only [IsColimit.fac,BinaryCofan.ι_app_right,eq_self_iff_true,BinaryCofan.mk_inr]
      /-
        🎉 no goals
      -/


/-- If `Y' ≅ Y`, then `X ⨿ Y` also is the coproduct of `X` and `Y'`. -/
noncomputable def BinaryCofan.isColimitCompRightIso {X Y Y' : C} (c : BinaryCofan X Y) (f : Y' ⟶ Y)
    [IsIso f] (h : IsColimit c) : IsColimit (BinaryCofan.mk c.inl (f ≫ c.inr)) :=
  BinaryCofan.isColimitFlip <| BinaryCofan.isColimitCompLeftIso _ f (BinaryCofan.isColimitFlip h)


/-- An abbreviation for `HasLimit (pair X Y)`. -/
abbrev HasBinaryProduct (X Y : C) :=
  HasLimit (pair X Y)


/-- An abbreviation for `HasColimit (pair X Y)`. -/
abbrev HasBinaryCoproduct (X Y : C) :=
  HasColimit (pair X Y)


/-- If we have a product of `X` and `Y`, we can access it using `prod X Y` or
    `X ⨯ Y`. -/
noncomputable abbrev prod (X Y : C) [HasBinaryProduct X Y] :=
  limit (pair X Y)


/-- If we have a coproduct of `X` and `Y`, we can access it using `coprod X Y` or
    `X ⨿ Y`. -/
noncomputable abbrev coprod (X Y : C) [HasBinaryCoproduct X Y] :=
  colimit (pair X Y)


/-- Notation for the product -/
notation:20 X " ⨯ " Y:20 => prod X Y


/-- Notation for the coproduct -/
notation:20 X " ⨿ " Y:20 => coprod X Y


/-- The projection map to the first component of the product. -/
noncomputable abbrev prod.fst {X Y : C} [HasBinaryProduct X Y] : X ⨯ Y ⟶ X :=
  limit.π (pair X Y) ⟨WalkingPair.left⟩


/-- The projection map to the second component of the product. -/
noncomputable abbrev prod.snd {X Y : C} [HasBinaryProduct X Y] : X ⨯ Y ⟶ Y :=
  limit.π (pair X Y) ⟨WalkingPair.right⟩


/-- The inclusion map from the first component of the coproduct. -/
noncomputable abbrev coprod.inl {X Y : C} [HasBinaryCoproduct X Y] : X ⟶ X ⨿ Y :=
  colimit.ι (pair X Y) ⟨WalkingPair.left⟩


/-- The inclusion map from the second component of the coproduct. -/
noncomputable abbrev coprod.inr {X Y : C} [HasBinaryCoproduct X Y] : Y ⟶ X ⨿ Y :=
  colimit.ι (pair X Y) ⟨WalkingPair.right⟩


/-- The binary fan constructed from the projection maps is a limit. -/
noncomputable def prodIsProd (X Y : C) [HasBinaryProduct X Y] :
    IsLimit (BinaryFan.mk (prod.fst : X ⨯ Y ⟶ X) prod.snd) :=
  (limit.isLimit _).ofIsoLimit (Cones.ext (Iso.refl _) (fun ⟨u⟩ => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y : C
      inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
      x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
      u : CategoryTheory.Limits.WalkingPair
      ⊢ Eq ((CategoryTheory.Limits.limit.cone (CategoryTheory.Limits.pair X Y)).π.ap …
    -/
    cases u
      /-
        case left
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y : C
        inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
        x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
        ⊢ Eq ((CategoryTheory.Limits.limit.cone (CategoryTheory.Limits.pair X Y)).π.ap …
      -/
    · dsimp; simp only [Category.id_comp]; rfl
                                           /-
                                             🎉 no goals
                                           -/
      /-
        case right
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y : C
        inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
        x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
        ⊢ Eq ((CategoryTheory.Limits.limit.cone (CategoryTheory.Limits.pair X Y)).π.ap …
      -/
    · dsimp; simp only [Category.id_comp]; rfl
                                           /-
                                             🎉 no goals
                                           -/
  ))


/-- The binary cofan constructed from the coprojection maps is a colimit. -/
noncomputable def coprodIsCoprod (X Y : C) [HasBinaryCoproduct X Y] :
    IsColimit (BinaryCofan.mk (coprod.inl : X ⟶ X ⨿ Y) coprod.inr) :=
  (colimit.isColimit _).ofIsoColimit (Cocones.ext (Iso.refl _) (fun ⟨u⟩ => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X✝ Y✝ X Y : C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
      x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
      u : CategoryTheory.Limits.WalkingPair
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
    -/
    cases u
      /-
        case left
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y : C
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
        x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
      -/
    · dsimp; simp only [Category.comp_id]
             /-
               🎉 no goals
             -/
      /-
        case right
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X✝ Y✝ X Y : C
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
        x✝ : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.colimit.cocon …
      -/
    · dsimp; simp only [Category.comp_id]
             /-
               🎉 no goals
             -/
  ))


@[ext 1100]
theorem prod.hom_ext {W X Y : C} [HasBinaryProduct X Y] {f g : W ⟶ X ⨯ Y}
    (h₁ : f ≫ prod.fst = g ≫ prod.fst) (h₂ : f ≫ prod.snd = g ≫ prod.snd) : f = g :=
  BinaryFan.IsLimit.hom_ext (limit.isLimit _) h₁ h₂


@[ext 1100]
theorem coprod.hom_ext {W X Y : C} [HasBinaryCoproduct X Y] {f g : X ⨿ Y ⟶ W}
    (h₁ : coprod.inl ≫ f = coprod.inl ≫ g) (h₂ : coprod.inr ≫ f = coprod.inr ≫ g) : f = g :=
  BinaryCofan.IsColimit.hom_ext (colimit.isColimit _) h₁ h₂


/-- If the product of `X` and `Y` exists, then every pair of morphisms `f : W ⟶ X` and `g : W ⟶ Y`
    induces a morphism `prod.lift f g : W ⟶ X ⨯ Y`. -/
noncomputable abbrev prod.lift {W X Y : C} [HasBinaryProduct X Y]
    (f : W ⟶ X) (g : W ⟶ Y) : W ⟶ X ⨯ Y :=
  limit.lift _ (BinaryFan.mk f g)


/-- diagonal arrow of the binary product in the category `fam I` -/
noncomputable abbrev diag (X : C) [HasBinaryProduct X X] : X ⟶ X ⨯ X :=
  prod.lift (𝟙 _) (𝟙 _)


/-- If the coproduct of `X` and `Y` exists, then every pair of morphisms `f : X ⟶ W` and
    `g : Y ⟶ W` induces a morphism `coprod.desc f g : X ⨿ Y ⟶ W`. -/
noncomputable abbrev coprod.desc {W X Y : C} [HasBinaryCoproduct X Y]
    (f : X ⟶ W) (g : Y ⟶ W) : X ⨿ Y ⟶ W :=
  colimit.desc _ (BinaryCofan.mk f g)


/-- codiagonal arrow of the binary coproduct -/
noncomputable abbrev codiag (X : C) [HasBinaryCoproduct X X] : X ⨿ X ⟶ X :=
  coprod.desc (𝟙 _) (𝟙 _)


@[reassoc]
theorem prod.lift_fst {W X Y : C} [HasBinaryProduct X Y] (f : W ⟶ X) (g : W ⟶ Y) :
    prod.lift f g ≫ prod.fst = f :=
  limit.lift_π _ _


@[reassoc]
theorem prod.lift_snd {W X Y : C} [HasBinaryProduct X Y] (f : W ⟶ X) (g : W ⟶ Y) :
    prod.lift f g ≫ prod.snd = g :=
  limit.lift_π _ _

-- The simp linter says simp can prove the reassoc version of this lemma.
-- Porting note: it can also prove the og version

@[reassoc]
theorem coprod.inl_desc {W X Y : C} [HasBinaryCoproduct X Y] (f : X ⟶ W) (g : Y ⟶ W) :
    coprod.inl ≫ coprod.desc f g = f :=
  colimit.ι_desc _ _

-- The simp linter says simp can prove the reassoc version of this lemma.
-- Porting note: it can also prove the og version

@[reassoc]
theorem coprod.inr_desc {W X Y : C} [HasBinaryCoproduct X Y] (f : X ⟶ W) (g : Y ⟶ W) :
    coprod.inr ≫ coprod.desc f g = g :=
  colimit.ι_desc _ _


instance prod.mono_lift_of_mono_left {W X Y : C} [HasBinaryProduct X Y] (f : W ⟶ X) (g : W ⟶ Y)
    [Mono f] : Mono (prod.lift f g) :=
  mono_of_mono_fac <| prod.lift_fst _ _


instance prod.mono_lift_of_mono_right {W X Y : C} [HasBinaryProduct X Y] (f : W ⟶ X) (g : W ⟶ Y)
    [Mono g] : Mono (prod.lift f g) :=
  mono_of_mono_fac <| prod.lift_snd _ _


instance coprod.epi_desc_of_epi_left {W X Y : C} [HasBinaryCoproduct X Y] (f : X ⟶ W) (g : Y ⟶ W)
    [Epi f] : Epi (coprod.desc f g) :=
  epi_of_epi_fac <| coprod.inl_desc _ _


instance coprod.epi_desc_of_epi_right {W X Y : C} [HasBinaryCoproduct X Y] (f : X ⟶ W) (g : Y ⟶ W)
    [Epi g] : Epi (coprod.desc f g) :=
  epi_of_epi_fac <| coprod.inr_desc _ _


/-- If the product of `X` and `Y` exists, then every pair of morphisms `f : W ⟶ X` and `g : W ⟶ Y`
    induces a morphism `l : W ⟶ X ⨯ Y` satisfying `l ≫ Prod.fst = f` and `l ≫ Prod.snd = g`. -/
noncomputable def prod.lift' {W X Y : C} [HasBinaryProduct X Y] (f : W ⟶ X) (g : W ⟶ Y) :
    { l : W ⟶ X ⨯ Y // l ≫ prod.fst = f ∧ l ≫ prod.snd = g } :=
  ⟨prod.lift f g, prod.lift_fst _ _, prod.lift_snd _ _⟩


/-- If the coproduct of `X` and `Y` exists, then every pair of morphisms `f : X ⟶ W` and
    `g : Y ⟶ W` induces a morphism `l : X ⨿ Y ⟶ W` satisfying `coprod.inl ≫ l = f` and
    `coprod.inr ≫ l = g`. -/
noncomputable def coprod.desc' {W X Y : C} [HasBinaryCoproduct X Y] (f : X ⟶ W) (g : Y ⟶ W) :
    { l : X ⨿ Y ⟶ W // coprod.inl ≫ l = f ∧ coprod.inr ≫ l = g } :=
  ⟨coprod.desc f g, coprod.inl_desc _ _, coprod.inr_desc _ _⟩


/-- If the products `W ⨯ X` and `Y ⨯ Z` exist, then every pair of morphisms `f : W ⟶ Y` and
    `g : X ⟶ Z` induces a morphism `prod.map f g : W ⨯ X ⟶ Y ⨯ Z`. -/
noncomputable def prod.map {W X Y Z : C} [HasBinaryProduct W X] [HasBinaryProduct Y Z]
    (f : W ⟶ Y) (g : X ⟶ Z) : W ⨯ X ⟶ Y ⨯ Z :=
  limMap (mapPair f g)


/-- If the coproducts `W ⨿ X` and `Y ⨿ Z` exist, then every pair of morphisms `f : W ⟶ Y` and
    `g : W ⟶ Z` induces a morphism `coprod.map f g : W ⨿ X ⟶ Y ⨿ Z`. -/
noncomputable def coprod.map {W X Y Z : C} [HasBinaryCoproduct W X] [HasBinaryCoproduct Y Z]
    (f : W ⟶ Y) (g : X ⟶ Z) : W ⨿ X ⟶ Y ⨿ Z :=
  colimMap (mapPair f g)


@[reassoc, simp]
theorem prod.comp_lift {V W X Y : C} [HasBinaryProduct X Y] (f : V ⟶ W) (g : W ⟶ X) (h : W ⟶ Y) :
                                                        /-
                                                          C : Type u
                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                          V W X Y : C
                                                          inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
                                                          f : Quiver.Hom V W
                                                          g : Quiver.Hom W X
                                                          h : Quiver.Hom W Y
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.prod.lift g  …
                                                        -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    f ≫ prod.lift g h = prod.lift (f ≫ g) (f ≫ h) := by ext <;> simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem prod.comp_diag {X Y : C} [HasBinaryProduct Y Y] (f : X ⟶ Y) :
                                     /-
                                       C : Type u
                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                       X Y : C
                                       inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
                                       f : Quiver.Hom X Y
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.diag Y)) (Ca …
                                     -/
    f ≫ diag Y = prod.lift f f := by simp
                                     /-
                                       🎉 no goals
                                     -/


@[reassoc (attr := simp)]
theorem prod.map_fst {W X Y Z : C} [HasBinaryProduct W X] [HasBinaryProduct Y Z] (f : W ⟶ Y)
    (g : X ⟶ Z) : prod.map f g ≫ prod.fst = prod.fst ≫ f :=
  limMap_π _ _


@[reassoc (attr := simp)]
theorem prod.map_snd {W X Y Z : C} [HasBinaryProduct W X] [HasBinaryProduct Y Z] (f : W ⟶ Y)
    (g : X ⟶ Z) : prod.map f g ≫ prod.snd = prod.snd ≫ g :=
  limMap_π _ _


@[simp]
theorem prod.map_id_id {X Y : C} [HasBinaryProduct X Y] : prod.map (𝟙 X) (𝟙 Y) = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
    ⊢ Eq (CategoryTheory.Limits.prod.map (CategoryTheory.CategoryStruct.id X) (Cat …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem prod.lift_fst_snd {X Y : C} [HasBinaryProduct X Y] :
                                                  /-
                                                    C : Type u
                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                    X Y : C
                                                    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Y
                                                    ⊢ Eq (CategoryTheory.Limits.prod.lift CategoryTheory.Limits.prod.fst CategoryT …
                                                  -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    prod.lift prod.fst prod.snd = 𝟙 (X ⨯ Y) := by ext <;> simp
                                                          /-
                                                            🎉 no goals
                                                          -/


@[reassoc (attr := simp)]
theorem prod.lift_map {V W X Y Z : C} [HasBinaryProduct W X] [HasBinaryProduct Y Z] (f : V ⟶ W)
    (g : V ⟶ X) (h : W ⟶ Y) (k : X ⟶ Z) :
                                                                   /-
                                                                     C : Type u
                                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                                     V W X Y Z : C
                                                                     inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W X
                                                                     inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Z
                                                                     f : Quiver.Hom V W
                                                                     g : Quiver.Hom V X
                                                                     h : Quiver.Hom W Y
                                                                     k : Quiver.Hom X Z
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift f g) …
                                                                   -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    prod.lift f g ≫ prod.map h k = prod.lift (f ≫ h) (g ≫ k) := by ext <;> simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp]
theorem prod.lift_fst_comp_snd_comp {W X Y Z : C} [HasBinaryProduct W Y] [HasBinaryProduct X Z]
    (g : W ⟶ X) (g' : Y ⟶ Z) : prod.lift (prod.fst ≫ g) (prod.snd ≫ g') = prod.map g g' := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W Y
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Z
    g : Quiver.Hom W X
    g' : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.Limits.prod.lift (CategoryTheory.CategoryStruct.comp Cate …
  -/
  rw [← prod.lift_map]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W Y
    inst✝ : CategoryTheory.Limits.HasBinaryProduct X Z
    g : Quiver.Hom W X
    g' : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift Cate …
  -/
  simp
  /-
    🎉 no goals
  -/

-- We take the right hand side here to be simp normal form, as this way composition lemmas for
-- `f ≫ h` and `g ≫ k` can fire (eg `id_comp`) , while `map_fst` and `map_snd` can still work just
-- as well.

@[reassoc (attr := simp)]
theorem prod.map_map {A₁ A₂ A₃ B₁ B₂ B₃ : C} [HasBinaryProduct A₁ B₁] [HasBinaryProduct A₂ B₂]
    [HasBinaryProduct A₃ B₃] (f : A₁ ⟶ A₂) (g : B₁ ⟶ B₂) (h : A₂ ⟶ A₃) (k : B₂ ⟶ B₃) :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                                                   A₁ A₂ A₃ B₁ B₂ B₃ : C
                                                                   inst✝² : CategoryTheory.Limits.HasBinaryProduct A₁ B₁
                                                                   inst✝¹ : CategoryTheory.Limits.HasBinaryProduct A₂ B₂
                                                                   inst✝ : CategoryTheory.Limits.HasBinaryProduct A₃ B₃
                                                                   f : Quiver.Hom A₁ A₂
                                                                   g : Quiver.Hom B₁ B₂
                                                                   h : Quiver.Hom A₂ A₃
                                                                   k : Quiver.Hom B₂ B₃
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.map f g)  …
                                                                 -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    prod.map f g ≫ prod.map h k = prod.map (f ≫ h) (g ≫ k) := by ext <;> simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

-- TODO: is it necessary to weaken the assumption here?

@[reassoc]
theorem prod.map_swap {A B X Y : C} (f : A ⟶ B) (g : X ⟶ Y)
    [HasLimitsOfShape (Discrete WalkingPair) C] :
                                                                                    /-
                                                                                      C : Type u
                                                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                      A B X Y : C
                                                                                      f : Quiver.Hom A B
                                                                                      g : Quiver.Hom X Y
                                                                                      inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete Catego …
                                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.map (Cate …
                                                                                    -/
    prod.map (𝟙 X) f ≫ prod.map g (𝟙 B) = prod.map g (𝟙 A) ≫ prod.map (𝟙 Y) f := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[reassoc]
theorem prod.map_comp_id {X Y Z W : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasBinaryProduct X W]
    [HasBinaryProduct Z W] [HasBinaryProduct Y W] :
                                                                       /-
                                                                         C : Type u
                                                                         inst✝³ : CategoryTheory.Category.{v, u} C
                                                                         X Y Z W : C
                                                                         f : Quiver.Hom X Y
                                                                         g : Quiver.Hom Y Z
                                                                         inst✝² : CategoryTheory.Limits.HasBinaryProduct X W
                                                                         inst✝¹ : CategoryTheory.Limits.HasBinaryProduct Z W
                                                                         inst✝ : CategoryTheory.Limits.HasBinaryProduct Y W
                                                                         ⊢ Eq (CategoryTheory.Limits.prod.map (CategoryTheory.CategoryStruct.comp f g)  …
                                                                       -/
    prod.map (f ≫ g) (𝟙 W) = prod.map f (𝟙 W) ≫ prod.map g (𝟙 W) := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[reassoc]
theorem prod.map_id_comp {X Y Z W : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasBinaryProduct W X]
    [HasBinaryProduct W Y] [HasBinaryProduct W Z] :
                                                                       /-
                                                                         C : Type u
                                                                         inst✝³ : CategoryTheory.Category.{v, u} C
                                                                         X Y Z W : C
                                                                         f : Quiver.Hom X Y
                                                                         g : Quiver.Hom Y Z
                                                                         inst✝² : CategoryTheory.Limits.HasBinaryProduct W X
                                                                         inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W Y
                                                                         inst✝ : CategoryTheory.Limits.HasBinaryProduct W Z
                                                                         ⊢ Eq (CategoryTheory.Limits.prod.map (CategoryTheory.CategoryStruct.id W) (Cat …
                                                                       -/
    prod.map (𝟙 W) (f ≫ g) = prod.map (𝟙 W) f ≫ prod.map (𝟙 W) g := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- If the products `W ⨯ X` and `Y ⨯ Z` exist, then every pair of isomorphisms `f : W ≅ Y` and
    `g : X ≅ Z` induces an isomorphism `prod.mapIso f g : W ⨯ X ≅ Y ⨯ Z`. -/
@[simps]
def prod.mapIso {W X Y Z : C} [HasBinaryProduct W X] [HasBinaryProduct Y Z] (f : W ≅ Y)
    (g : X ≅ Z) : W ⨯ X ≅ Y ⨯ Z where
  hom := prod.map f.hom g.hom
  inv := prod.map f.inv g.inv


instance isIso_prod {W X Y Z : C} [HasBinaryProduct W X] [HasBinaryProduct Y Z] (f : W ⟶ Y)
    (g : X ⟶ Z) [IsIso f] [IsIso g] : IsIso (prod.map f g) :=
  (prod.mapIso (asIso f) (asIso g)).isIso_hom


instance prod.map_mono {C : Type*} [Category C] {W X Y Z : C} (f : W ⟶ Y) (g : X ⟶ Z) [Mono f]
    [Mono g] [HasBinaryProduct W X] [HasBinaryProduct Y Z] : Mono (prod.map f g) :=
  ⟨fun i₁ i₂ h => by
    /-
      C✝ : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C✝
      X✝ Y✝ : C✝
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      W X Y Z : C
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      inst✝³ : CategoryTheory.Mono f
      inst✝² : CategoryTheory.Mono g
      inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W X
      inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Z
      Z✝ : C
      i₁ i₂ : Quiver.Hom Z✝ (CategoryTheory.Limits.prod W X)
      h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.map  …
      ⊢ Eq i₁ i₂
    -/
    ext
      /-
        case h₁
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Mono f
        inst✝² : CategoryTheory.Mono g
        inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom Z✝ (CategoryTheory.Limits.prod W X)
        h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.map  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i₁ CategoryTheory.Limits.prod.fst) (C …
      -/
    · rw [← cancel_mono f]
      /-
        case h₁
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Mono f
        inst✝² : CategoryTheory.Mono g
        inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom Z✝ (CategoryTheory.Limits.prod W X)
        h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.map  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
      -/
      simpa using congr_arg (fun f => f ≫ prod.fst) h
      /-
        🎉 no goals
      -/
      /-
        case h₂
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Mono f
        inst✝² : CategoryTheory.Mono g
        inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom Z✝ (CategoryTheory.Limits.prod W X)
        h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.map  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp i₁ CategoryTheory.Limits.prod.snd) (C …
      -/
    · rw [← cancel_mono g]
      /-
        case h₂
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Mono f
        inst✝² : CategoryTheory.Mono g
        inst✝¹ : CategoryTheory.Limits.HasBinaryProduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom Z✝ (CategoryTheory.Limits.prod W X)
        h : Eq (CategoryTheory.CategoryStruct.comp i₁ (CategoryTheory.Limits.prod.map  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
      -/
      simpa using congr_arg (fun f => f ≫ prod.snd) h⟩
      /-
        🎉 no goals
      -/


@[reassoc]
theorem prod.diag_map {X Y : C} (f : X ⟶ Y) [HasBinaryProduct X X] [HasBinaryProduct Y Y] :
                                             /-
                                               C : Type u
                                               inst✝² : CategoryTheory.Category.{v, u} C
                                               X Y : C
                                               f : Quiver.Hom X Y
                                               inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X X
                                               inst✝ : CategoryTheory.Limits.HasBinaryProduct Y Y
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diag X) (Categ …
                                             -/
    diag X ≫ prod.map f f = f ≫ diag Y := by simp
                                             /-
                                               🎉 no goals
                                             -/


@[reassoc]
theorem prod.diag_map_fst_snd {X Y : C} [HasBinaryProduct X Y] [HasBinaryProduct (X ⨯ Y) (X ⨯ Y)] :
                                                                /-
                                                                  C : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                  X Y : C
                                                                  inst✝¹ : CategoryTheory.Limits.HasBinaryProduct X Y
                                                                  inst✝ : CategoryTheory.Limits.HasBinaryProduct (CategoryTheory.Limits.prod X Y …
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diag (Category …
                                                                -/
    diag (X ⨯ Y) ≫ prod.map prod.fst prod.snd = 𝟙 (X ⨯ Y) := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[reassoc]
theorem prod.diag_map_fst_snd_comp [HasLimitsOfShape (Discrete WalkingPair) C] {X X' Y Y' : C}
    (g : X ⟶ Y) (g' : X' ⟶ Y') :
                                                                                  /-
                                                                                    C : Type u
                                                                                    inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                    inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete Catego …
                                                                                    X X' Y Y' : C
                                                                                    g : Quiver.Hom X Y
                                                                                    g' : Quiver.Hom X' Y'
                                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diag (Category …
                                                                                  -/
    diag (X ⨯ X') ≫ prod.map (prod.fst ≫ g) (prod.snd ≫ g') = prod.map g g' := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance {X : C} [HasBinaryProduct X X] : IsSplitMono (diag X) :=
  IsSplitMono.mk' { retraction := prod.fst }


@[simp] -- Porting note: removing reassoc tag since result is not hygienic (two h's)
theorem coprod.desc_comp {V W X Y : C} [HasBinaryCoproduct X Y] (f : V ⟶ W) (g : X ⟶ V)
    (h : Y ⟶ V) : coprod.desc g h ≫ f = coprod.desc (g ≫ f) (h ≫ f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    V W X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    f : Quiver.Hom V W
    g : Quiver.Hom X V
    h : Quiver.Hom Y V
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc g  …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/

-- Porting note: hand generated reassoc here. Simp can prove it

theorem coprod.desc_comp_assoc {C : Type u} [Category C] {V W X Y : C}
    [HasBinaryCoproduct X Y] (f : V ⟶ W) (g : X ⟶ V) (h : Y ⟶ V) {Z : C} (l : W ⟶ Z) :
                                                                    /-
                                                                      C : Type u
                                                                      inst✝¹ : CategoryTheory.Category.{u_1, u} C
                                                                      V W X Y : C
                                                                      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
                                                                      f : Quiver.Hom V W
                                                                      g : Quiver.Hom X V
                                                                      h : Quiver.Hom Y V
                                                                      Z : C
                                                                      l : Quiver.Hom W Z
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc g  …
                                                                    -/
    coprod.desc g h ≫ f ≫ l = coprod.desc (g ≫ f) (h ≫ f) ≫ l := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem coprod.diag_comp {X Y : C} [HasBinaryCoproduct X X] (f : X ⟶ Y) :
                                         /-
                                           C : Type u
                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                           X Y : C
                                           inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X X
                                           f : Quiver.Hom X Y
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.codiag X) f) ( …
                                         -/
    codiag X ≫ f = coprod.desc f f := by simp
                                         /-
                                           🎉 no goals
                                         -/


@[reassoc (attr := simp)]
theorem coprod.inl_map {W X Y Z : C} [HasBinaryCoproduct W X] [HasBinaryCoproduct Y Z] (f : W ⟶ Y)
    (g : X ⟶ Z) : coprod.inl ≫ coprod.map f g = f ≫ coprod.inl :=
  ι_colimMap _ _


@[reassoc (attr := simp)]
theorem coprod.inr_map {W X Y Z : C} [HasBinaryCoproduct W X] [HasBinaryCoproduct Y Z] (f : W ⟶ Y)
    (g : X ⟶ Z) : coprod.inr ≫ coprod.map f g = g ≫ coprod.inr :=
  ι_colimMap _ _


@[simp]
theorem coprod.map_id_id {X Y : C} [HasBinaryCoproduct X Y] : coprod.map (𝟙 X) (𝟙 Y) = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
    ⊢ Eq (CategoryTheory.Limits.coprod.map (CategoryTheory.CategoryStruct.id X) (C …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem coprod.desc_inl_inr {X Y : C} [HasBinaryCoproduct X Y] :
                                                        /-
                                                          C : Type u
                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                          X Y : C
                                                          inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Y
                                                          ⊢ Eq (CategoryTheory.Limits.coprod.desc CategoryTheory.Limits.coprod.inl Categ …
                                                        -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
    coprod.desc coprod.inl coprod.inr = 𝟙 (X ⨿ Y) := by ext <;> simp
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- The simp linter says simp can prove the reassoc version of this lemma.

@[reassoc, simp]
theorem coprod.map_desc {S T U V W : C} [HasBinaryCoproduct U W] [HasBinaryCoproduct T V]
    (f : U ⟶ S) (g : W ⟶ S) (h : T ⟶ U) (k : V ⟶ W) :
    coprod.map h k ≫ coprod.desc f g = coprod.desc (h ≫ f) (k ≫ g) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    S T U V W : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct U W
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct T V
    f : Quiver.Hom U S
    g : Quiver.Hom W S
    h : Quiver.Hom T U
    k : Quiver.Hom V W
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map h k …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


@[simp]
theorem coprod.desc_comp_inl_comp_inr {W X Y Z : C} [HasBinaryCoproduct W Y]
    [HasBinaryCoproduct X Z] (g : W ⟶ X) (g' : Y ⟶ Z) :
    coprod.desc (g ≫ coprod.inl) (g' ≫ coprod.inr) = coprod.map g g' := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W Y
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X Z
    g : Quiver.Hom W X
    g' : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.Limits.coprod.desc (CategoryTheory.CategoryStruct.comp g  …
  -/
  rw [← coprod.map_desc]; simp
                          /-
                            🎉 no goals
                          -/

-- We take the right hand side here to be simp normal form, as this way composition lemmas for
-- `f ≫ h` and `g ≫ k` can fire (eg `id_comp`) , while `inl_map` and `inr_map` can still work just
-- as well.

@[reassoc (attr := simp)]
theorem coprod.map_map {A₁ A₂ A₃ B₁ B₂ B₃ : C} [HasBinaryCoproduct A₁ B₁] [HasBinaryCoproduct A₂ B₂]
    [HasBinaryCoproduct A₃ B₃] (f : A₁ ⟶ A₂) (g : B₁ ⟶ B₂) (h : A₂ ⟶ A₃) (k : B₂ ⟶ B₃) :
    coprod.map f g ≫ coprod.map h k = coprod.map (f ≫ h) (g ≫ k) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    A₁ A₂ A₃ B₁ B₂ B₃ : C
    inst✝² : CategoryTheory.Limits.HasBinaryCoproduct A₁ B₁
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct A₂ B₂
    inst✝ : CategoryTheory.Limits.HasBinaryCoproduct A₃ B₃
    f : Quiver.Hom A₁ A₂
    g : Quiver.Hom B₁ B₂
    h : Quiver.Hom A₂ A₃
    k : Quiver.Hom B₂ B₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f g …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/

-- I don't think it's a good idea to make any of the following three simp lemmas.

@[reassoc]
theorem coprod.map_swap {A B X Y : C} (f : A ⟶ B) (g : X ⟶ Y)
    [HasColimitsOfShape (Discrete WalkingPair) C] :
                                                                                            /-
                                                                                              C : Type u
                                                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                              A B X Y : C
                                                                                              f : Quiver.Hom A B
                                                                                              g : Quiver.Hom X Y
                                                                                              inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete Cate …
                                                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map (Ca …
                                                                                            -/
    coprod.map (𝟙 X) f ≫ coprod.map g (𝟙 B) = coprod.map g (𝟙 A) ≫ coprod.map (𝟙 Y) f := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[reassoc]
theorem coprod.map_comp_id {X Y Z W : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasBinaryCoproduct Z W]
    [HasBinaryCoproduct Y W] [HasBinaryCoproduct X W] :
                                                                             /-
                                                                               C : Type u
                                                                               inst✝³ : CategoryTheory.Category.{v, u} C
                                                                               X Y Z W : C
                                                                               f : Quiver.Hom X Y
                                                                               g : Quiver.Hom Y Z
                                                                               inst✝² : CategoryTheory.Limits.HasBinaryCoproduct Z W
                                                                               inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct Y W
                                                                               inst✝ : CategoryTheory.Limits.HasBinaryCoproduct X W
                                                                               ⊢ Eq (CategoryTheory.Limits.coprod.map (CategoryTheory.CategoryStruct.comp f g …
                                                                             -/
    coprod.map (f ≫ g) (𝟙 W) = coprod.map f (𝟙 W) ≫ coprod.map g (𝟙 W) := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[reassoc]
theorem coprod.map_id_comp {X Y Z W : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasBinaryCoproduct W X]
    [HasBinaryCoproduct W Y] [HasBinaryCoproduct W Z] :
                                                                             /-
                                                                               C : Type u
                                                                               inst✝³ : CategoryTheory.Category.{v, u} C
                                                                               X Y Z W : C
                                                                               f : Quiver.Hom X Y
                                                                               g : Quiver.Hom Y Z
                                                                               inst✝² : CategoryTheory.Limits.HasBinaryCoproduct W X
                                                                               inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W Y
                                                                               inst✝ : CategoryTheory.Limits.HasBinaryCoproduct W Z
                                                                               ⊢ Eq (CategoryTheory.Limits.coprod.map (CategoryTheory.CategoryStruct.id W) (C …
                                                                             -/
    coprod.map (𝟙 W) (f ≫ g) = coprod.map (𝟙 W) f ≫ coprod.map (𝟙 W) g := by simp
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- If the coproducts `W ⨿ X` and `Y ⨿ Z` exist, then every pair of isomorphisms `f : W ≅ Y` and
   `g : W ≅ Z` induces an isomorphism `coprod.mapIso f g : W ⨿ X ≅ Y ⨿ Z`. -/
@[simps]
def coprod.mapIso {W X Y Z : C} [HasBinaryCoproduct W X] [HasBinaryCoproduct Y Z] (f : W ≅ Y)
    (g : X ≅ Z) : W ⨿ X ≅ Y ⨿ Z where
  hom := coprod.map f.hom g.hom
  inv := coprod.map f.inv g.inv


instance isIso_coprod {W X Y Z : C} [HasBinaryCoproduct W X] [HasBinaryCoproduct Y Z] (f : W ⟶ Y)
    (g : X ⟶ Z) [IsIso f] [IsIso g] : IsIso (coprod.map f g) :=
  (coprod.mapIso (asIso f) (asIso g)).isIso_hom


instance coprod.map_epi {C : Type*} [Category C] {W X Y Z : C} (f : W ⟶ Y) (g : X ⟶ Z) [Epi f]
    [Epi g] [HasBinaryCoproduct W X] [HasBinaryCoproduct Y Z] : Epi (coprod.map f g) :=
  ⟨fun i₁ i₂ h => by
    /-
      C✝ : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C✝
      X✝ Y✝ : C✝
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      W X Y Z : C
      f : Quiver.Hom W Y
      g : Quiver.Hom X Z
      inst✝³ : CategoryTheory.Epi f
      inst✝² : CategoryTheory.Epi g
      inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W X
      inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
      Z✝ : C
      i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) Z✝
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f …
      ⊢ Eq i₁ i₂
    -/
    ext
      /-
        case h₁
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Epi f
        inst✝² : CategoryTheory.Epi g
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) Z✝
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl i₁)  …
      -/
    · rw [← cancel_epi f]
      /-
        case h₁
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Epi f
        inst✝² : CategoryTheory.Epi g
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) Z✝
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      simpa using congr_arg (fun f => coprod.inl ≫ f) h
      /-
        🎉 no goals
      -/
      /-
        case h₂
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Epi f
        inst✝² : CategoryTheory.Epi g
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) Z✝
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr i₁)  …
      -/
    · rw [← cancel_epi g]
      /-
        case h₂
        C✝ : Type u
        inst✝⁵ : CategoryTheory.Category.{v, u} C✝
        X✝ Y✝ : C✝
        C : Type u_1
        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
        W X Y Z : C
        f : Quiver.Hom W Y
        g : Quiver.Hom X Z
        inst✝³ : CategoryTheory.Epi f
        inst✝² : CategoryTheory.Epi g
        inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct W X
        inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Z
        Z✝ : C
        i₁ i₂ : Quiver.Hom (CategoryTheory.Limits.coprod Y Z) Z✝
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
      -/
      simpa using congr_arg (fun f => coprod.inr ≫ f) h⟩
      /-
        🎉 no goals
      -/

-- The simp linter says simp can prove the reassoc version of this lemma.
-- Porting note: and the og version too

@[reassoc]
theorem coprod.map_codiag {X Y : C} (f : X ⟶ Y) [HasBinaryCoproduct X X] [HasBinaryCoproduct Y Y] :
                                                   /-
                                                     C : Type u
                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                     X Y : C
                                                     f : Quiver.Hom X Y
                                                     inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct X X
                                                     inst✝ : CategoryTheory.Limits.HasBinaryCoproduct Y Y
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map f f …
                                                   -/
    coprod.map f f ≫ codiag Y = codiag X ≫ f := by simp
                                                   /-
                                                     🎉 no goals
                                                   -/

-- The simp linter says simp can prove the reassoc version of this lemma.
-- Porting note: and the og version too

@[reassoc]
theorem coprod.map_inl_inr_codiag {X Y : C} [HasBinaryCoproduct X Y]
    [HasBinaryCoproduct (X ⨿ Y) (X ⨿ Y)] :
                                                                        /-
                                                                          C : Type u
                                                                          inst✝² : CategoryTheory.Category.{v, u} C
                                                                          X Y : C
                                                                          inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct X Y
                                                                          inst✝ : CategoryTheory.Limits.HasBinaryCoproduct (CategoryTheory.Limits.coprod …
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map Cat …
                                                                        -/
    coprod.map coprod.inl coprod.inr ≫ codiag (X ⨿ Y) = 𝟙 (X ⨿ Y) := by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

-- The simp linter says simp can prove the reassoc version of this lemma.
-- Porting note: and the og version too

@[reassoc]
theorem coprod.map_comp_inl_inr_codiag [HasColimitsOfShape (Discrete WalkingPair) C] {X X' Y Y' : C}
    (g : X ⟶ Y) (g' : X' ⟶ Y') :
                                                                                            /-
                                                                                              C : Type u
                                                                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                              inst✝ : CategoryTheory.Limits.HasColimitsOfShape (CategoryTheory.Discrete Cate …
                                                                                              X X' Y Y' : C
                                                                                              g : Quiver.Hom X Y
                                                                                              g' : Quiver.Hom X' Y'
                                                                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map (Ca …
                                                                                            -/
    coprod.map (g ≫ coprod.inl) (g' ≫ coprod.inr) ≫ codiag (Y ⨿ Y') = coprod.map g g' := by simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


/-- `HasBinaryProducts` represents a choice of product for every pair of objects.

See <https://stacks.math.columbia.edu/tag/001T>.
-/
abbrev HasBinaryProducts :=
  HasLimitsOfShape (Discrete WalkingPair) C


/-- `HasBinaryCoproducts` represents a choice of coproduct for every pair of objects.

See <https://stacks.math.columbia.edu/tag/04AP>.
-/
abbrev HasBinaryCoproducts :=
  HasColimitsOfShape (Discrete WalkingPair) C


/-- If `C` has all limits of diagrams `pair X Y`, then it has all binary products -/
theorem hasBinaryProducts_of_hasLimit_pair [∀ {X Y : C}, HasLimit (pair X Y)] :
    HasBinaryProducts C :=
  { has_limit := fun F => hasLimitOfIso (diagramIsoPair F).symm }


/-- If `C` has all colimits of diagrams `pair X Y`, then it has all binary coproducts -/
theorem hasBinaryCoproducts_of_hasColimit_pair [∀ {X Y : C}, HasColimit (pair X Y)] :
    HasBinaryCoproducts C :=
  { has_colimit := fun F => hasColimitOfIso (diagramIsoPair F) }


/-- The braiding isomorphism which swaps a binary product. -/
@[simps]
def prod.braiding (P Q : C) [HasBinaryProduct P Q] [HasBinaryProduct Q P] : P ⨯ Q ≅ Q ⨯ P where
  hom := prod.lift prod.snd prod.fst
  inv := prod.lift prod.snd prod.fst


/-- The braiding isomorphism can be passed through a map by swapping the order. -/
@[reassoc]
theorem braid_natural [HasBinaryProducts C] {W X Y Z : C} (f : X ⟶ Y) (g : Z ⟶ W) :
                                                                                          /-
                                                                                            C : Type u
                                                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                            inst✝ : CategoryTheory.Limits.HasBinaryProducts C
                                                                                            W X Y Z : C
                                                                                            f : Quiver.Hom X Y
                                                                                            g : Quiver.Hom Z W
                                                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.map f g)  …
                                                                                          -/
    prod.map f g ≫ (prod.braiding _ _).hom = (prod.braiding _ _).hom ≫ prod.map g f := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[reassoc]
theorem prod.symmetry' (P Q : C) [HasBinaryProduct P Q] [HasBinaryProduct Q P] :
    prod.lift prod.snd prod.fst ≫ prod.lift prod.snd prod.fst = 𝟙 (P ⨯ Q) :=
  (prod.braiding _ _).hom_inv_id


/-- The braiding isomorphism is symmetric. -/
@[reassoc]
theorem prod.symmetry (P Q : C) [HasBinaryProduct P Q] [HasBinaryProduct Q P] :
    (prod.braiding P Q).hom ≫ (prod.braiding Q P).hom = 𝟙 _ :=
  (prod.braiding _ _).hom_inv_id


/-- The associator isomorphism for binary products. -/
@[simps]
def prod.associator [HasBinaryProducts C] (P Q R : C) : (P ⨯ Q) ⨯ R ≅ P ⨯ Q ⨯ R where
  hom := prod.lift (prod.fst ≫ prod.fst) (prod.lift (prod.fst ≫ prod.snd) prod.snd)
  inv := prod.lift (prod.lift prod.fst (prod.snd ≫ prod.fst)) (prod.snd ≫ prod.snd)


@[reassoc]
theorem prod.pentagon [HasBinaryProducts C] (W X Y Z : C) :
    prod.map (prod.associator W X Y).hom (𝟙 Z) ≫
        (prod.associator W (X ⨯ Y) Z).hom ≫ prod.map (𝟙 W) (prod.associator X Y Z).hom =
      (prod.associator (W ⨯ X) Y Z).hom ≫ (prod.associator W X (Y ⨯ Z)).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.map (Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
theorem prod.associator_naturality [HasBinaryProducts C] {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : X₁ ⟶ Y₁)
    (f₂ : X₂ ⟶ Y₂) (f₃ : X₃ ⟶ Y₃) :
    prod.map (prod.map f₁ f₂) f₃ ≫ (prod.associator Y₁ Y₂ Y₃).hom =
      (prod.associator X₁ X₂ X₃).hom ≫ prod.map f₁ (prod.map f₂ f₃) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.map (Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The left unitor isomorphism for binary products with the terminal object. -/
@[simps]
def prod.leftUnitor (P : C) [HasBinaryProduct (⊤_ C) P] : (⊤_ C) ⨯ P ≅ P where
  hom := prod.snd
  inv := prod.lift (terminal.from P) (𝟙 _)
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasTerminal C
                     P : C
                     inst✝ : CategoryTheory.Limits.HasBinaryProduct (CategoryTheory.Limits.terminal …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.snd (Categ …
                   -/
                                          /-
                                            🎉 no goals
                                          -/
  hom_inv_id := by apply prod.hom_ext <;> simp [eq_iff_true_of_subsingleton]
                                          /-
                                            🎉 no goals
                                          -/
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasTerminal C
                     P : C
                     inst✝ : CategoryTheory.Limits.HasBinaryProduct (CategoryTheory.Limits.terminal …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
                   -/
  inv_hom_id := by simp
                   /-
                     🎉 no goals
                   -/


/-- The right unitor isomorphism for binary products with the terminal object. -/
@[simps]
def prod.rightUnitor (P : C) [HasBinaryProduct P (⊤_ C)] : P ⨯ ⊤_ C ≅ P where
  hom := prod.fst
  inv := prod.lift (𝟙 _) (terminal.from P)
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasTerminal C
                     P : C
                     inst✝ : CategoryTheory.Limits.HasBinaryProduct P (CategoryTheory.Limits.termin …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.prod.fst (Categ …
                   -/
                                          /-
                                            🎉 no goals
                                          -/
  hom_inv_id := by apply prod.hom_ext <;> simp [eq_iff_true_of_subsingleton]
                                          /-
                                            🎉 no goals
                                          -/
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasTerminal C
                     P : C
                     inst✝ : CategoryTheory.Limits.HasBinaryProduct P (CategoryTheory.Limits.termin …
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.lift (Cat …
                   -/
  inv_hom_id := by simp
                   /-
                     🎉 no goals
                   -/


@[reassoc]
theorem prod.leftUnitor_hom_naturality [HasBinaryProducts C] (f : X ⟶ Y) :
    prod.map (𝟙 _) f ≫ (prod.leftUnitor Y).hom = (prod.leftUnitor X).hom ≫ f :=
  prod.map_snd _ _


@[reassoc]
theorem prod.leftUnitor_inv_naturality [HasBinaryProducts C] (f : X ⟶ Y) :
    (prod.leftUnitor X).inv ≫ prod.map (𝟙 _) f = f ≫ (prod.leftUnitor Y).inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.leftUnito …
  -/
  rw [Iso.inv_comp_eq, ← Category.assoc, Iso.eq_comp_inv, prod.leftUnitor_hom_naturality]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem prod.rightUnitor_hom_naturality [HasBinaryProducts C] (f : X ⟶ Y) :
    prod.map f (𝟙 _) ≫ (prod.rightUnitor Y).hom = (prod.rightUnitor X).hom ≫ f :=
  prod.map_fst _ _


@[reassoc]
theorem prod_rightUnitor_inv_naturality [HasBinaryProducts C] (f : X ⟶ Y) :
    (prod.rightUnitor X).inv ≫ prod.map f (𝟙 _) = f ≫ (prod.rightUnitor Y).inv := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.rightUnit …
  -/
  rw [Iso.inv_comp_eq, ← Category.assoc, Iso.eq_comp_inv, prod.rightUnitor_hom_naturality]
  /-
    🎉 no goals
  -/


theorem prod.triangle [HasBinaryProducts C] (X Y : C) :
    (prod.associator X (⊤_ C) Y).hom ≫ prod.map (𝟙 X) (prod.leftUnitor Y).hom =
      prod.map (prod.rightUnitor X).hom (𝟙 Y) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasTerminal C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.prod.associato …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- The braiding isomorphism which swaps a binary coproduct. -/
@[simps]
def coprod.braiding (P Q : C) : P ⨿ Q ≅ Q ⨿ P where
  hom := coprod.desc coprod.inr coprod.inl
  inv := coprod.desc coprod.inr coprod.inl


@[reassoc]
theorem coprod.symmetry' (P Q : C) :
    coprod.desc coprod.inr coprod.inl ≫ coprod.desc coprod.inr coprod.inl = 𝟙 (P ⨿ Q) :=
  (coprod.braiding _ _).hom_inv_id


/-- The braiding isomorphism is symmetric. -/
theorem coprod.symmetry (P Q : C) : (coprod.braiding P Q).hom ≫ (coprod.braiding Q P).hom = 𝟙 _ :=
  coprod.symmetry' _ _


/-- The associator isomorphism for binary coproducts. -/
@[simps]
def coprod.associator (P Q R : C) : (P ⨿ Q) ⨿ R ≅ P ⨿ Q ⨿ R where
  hom := coprod.desc (coprod.desc coprod.inl (coprod.inl ≫ coprod.inr)) (coprod.inr ≫ coprod.inr)
  inv := coprod.desc (coprod.inl ≫ coprod.inl) (coprod.desc (coprod.inr ≫ coprod.inl) coprod.inr)


theorem coprod.pentagon (W X Y Z : C) :
    coprod.map (coprod.associator W X Y).hom (𝟙 Z) ≫
        (coprod.associator W (X ⨿ Y) Z).hom ≫ coprod.map (𝟙 W) (coprod.associator X Y Z).hom =
      (coprod.associator (W ⨿ X) Y Z).hom ≫ (coprod.associator W X (Y ⨿ Z)).hom := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    W X Y Z : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem coprod.associator_naturality {X₁ X₂ X₃ Y₁ Y₂ Y₃ : C} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂)
    (f₃ : X₃ ⟶ Y₃) :
    coprod.map (coprod.map f₁ f₂) f₃ ≫ (coprod.associator Y₁ Y₂ Y₃).hom =
      (coprod.associator X₁ X₂ X₃).hom ≫ coprod.map f₁ (coprod.map f₂ f₃) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    X₁ X₂ X₃ Y₁ Y₂ Y₃ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map (Ca …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The left unitor isomorphism for binary coproducts with the initial object. -/
@[simps]
def coprod.leftUnitor (P : C) : (⊥_ C) ⨿ P ≅ P where
  hom := coprod.desc (initial.to P) (𝟙 _)
  inv := coprod.inr
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                     inst✝ : CategoryTheory.Limits.HasInitial C
                     P : C
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc (C …
                   -/
                                            /-
                                              🎉 no goals
                                            -/
  hom_inv_id := by apply coprod.hom_ext <;> simp [eq_iff_true_of_subsingleton]
                                            /-
                                              🎉 no goals
                                            -/
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                     inst✝ : CategoryTheory.Limits.HasInitial C
                     P : C
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
                   -/
  inv_hom_id := by simp
                   /-
                     🎉 no goals
                   -/


/-- The right unitor isomorphism for binary coproducts with the initial object. -/
@[simps]
def coprod.rightUnitor (P : C) : P ⨿ ⊥_ C ≅ P where
  hom := coprod.desc (𝟙 _) (initial.to P)
  inv := coprod.inl
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                     inst✝ : CategoryTheory.Limits.HasInitial C
                     P : C
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.desc (C …
                   -/
                                            /-
                                              🎉 no goals
                                            -/
  hom_inv_id := by apply coprod.hom_ext <;> simp [eq_iff_true_of_subsingleton]
                                            /-
                                              🎉 no goals
                                            -/
                   /-
                     C : Type u
                     inst✝² : CategoryTheory.Category.{v, u} C
                     X Y : C
                     inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                     inst✝ : CategoryTheory.Limits.HasInitial C
                     P : C
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
                   -/
  inv_hom_id := by simp
                   /-
                     🎉 no goals
                   -/


theorem coprod.triangle (X Y : C) :
    (coprod.associator X (⊥_ C) Y).hom ≫ coprod.map (𝟙 X) (coprod.leftUnitor Y).hom =
      coprod.map (coprod.rightUnitor X).hom (𝟙 Y) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
    inst✝ : CategoryTheory.Limits.HasInitial C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.associa …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


/-- The binary product functor. -/
@[simps]
def prod.functor : C ⥤ C ⥤ C where
  obj X :=
    { obj := fun Y => X ⨯ Y
      map := fun {_ _} => prod.map (𝟙 X) }
  map f :=
    { app := fun T => prod.map f (𝟙 T) }


/-- The product functor can be decomposed. -/
def prod.functorLeftComp (X Y : C) :
    prod.functor.obj (X ⨯ Y) ≅ prod.functor.obj Y ⋙ prod.functor.obj X :=
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X✝ Y✝ : C
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasBinaryProducts C
    X Y : C
    ⊢ ∀ {X_1 Y_1 : C} (f : Quiver.Hom X_1 Y_1), Eq (CategoryTheory.CategoryStruct. …
  -/
  NatIso.ofComponents (prod.associator _ _)
  /-
    🎉 no goals
  -/


/-- The binary coproduct functor. -/
@[simps]
def coprod.functor : C ⥤ C ⥤ C where
  obj X :=
    { obj := fun Y => X ⨿ Y
      map := fun {_ _} => coprod.map (𝟙 X) }
  map f := { app := fun T => coprod.map f (𝟙 T) }


/-- The coproduct functor can be decomposed. -/
def coprod.functorLeftComp (X Y : C) :
    coprod.functor.obj (X ⨿ Y) ≅ coprod.functor.obj Y ⋙ coprod.functor.obj X :=
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X✝ Y✝ : C
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
    X Y : C
    ⊢ ∀ {X_1 Y_1 : C} (f : Quiver.Hom X_1 Y_1), Eq (CategoryTheory.CategoryStruct. …
  -/
  NatIso.ofComponents (coprod.associator _ _)
  /-
    🎉 no goals
  -/


/-- The product comparison morphism.

In `CategoryTheory/Limits/Preserves` we show this is always an iso iff F preserves binary products.
-/
def prodComparison (F : C ⥤ D) (A B : C) [HasBinaryProduct A B]
    [HasBinaryProduct (F.obj A) (F.obj B)] : F.obj (A ⨯ B) ⟶ F.obj A ⨯ F.obj B :=
  prod.lift (F.map prod.fst) (F.map prod.snd)


@[reassoc (attr := simp)]
theorem prodComparison_fst : prodComparison F A B ≫ prod.fst = F.map prod.fst :=
  prod.lift_fst _ _


@[reassoc (attr := simp)]
theorem prodComparison_snd : prodComparison F A B ≫ prod.snd = F.map prod.snd :=
  prod.lift_snd _ _


/-- Naturality of the `prodComparison` morphism in both arguments. -/
@[reassoc]
theorem prodComparison_natural (f : A ⟶ A') (g : B ⟶ B') :
    F.map (prod.map f g) ≫ prodComparison F A' B' =
      prodComparison F A B ≫ prod.map (F.map f) (F.map g) := by
  rw [prodComparison, prodComparison, prod.lift_map, ← F.map_comp, ← F.map_comp, prod.comp_lift, ←
    F.map_comp, prod.map_fst, ← F.map_comp, prod.map_snd]


/-- The product comparison morphism from `F(A ⨯ -)` to `FA ⨯ F-`, whose components are given by
`prodComparison`.
-/
@[simps]
def prodComparisonNatTrans [HasBinaryProducts C] [HasBinaryProducts D] (F : C ⥤ D) (A : C) :
    prod.functor.obj A ⋙ F ⟶ F ⋙ prod.functor.obj (F.obj A) where
  app B := prodComparison F A B
                     /-
                       C : Type u
                       inst✝¹⁰ : CategoryTheory.Category.{v, u} C
                       X Y : C
                       D : Type u₂
                       inst✝⁹ : CategoryTheory.Category.{w, u₂} D
                       E : Type u₃
                       inst✝⁸ : CategoryTheory.Category.{w', u₃} E
                       F✝ : CategoryTheory.Functor C D
                       G : CategoryTheory.Functor D E
                       A✝ A' B B' : C
                       inst✝⁷ : CategoryTheory.Limits.HasBinaryProduct A✝ B
                       inst✝⁶ : CategoryTheory.Limits.HasBinaryProduct A' B'
                       inst✝⁵ : CategoryTheory.Limits.HasBinaryProduct (F✝.obj A✝) (F✝.obj B)
                       inst✝⁴ : CategoryTheory.Limits.HasBinaryProduct (F✝.obj A') (F✝.obj B')
                       inst✝³ : CategoryTheory.Limits.HasBinaryProduct (G.obj (F✝.obj A✝)) (G.obj (F✝ …
                       inst✝² : CategoryTheory.Limits.HasBinaryProduct ((F✝.comp G).obj A✝) ((F✝.comp …
                       inst✝¹ : CategoryTheory.Limits.HasBinaryProducts C
                       inst✝ : CategoryTheory.Limits.HasBinaryProducts D
                       F : CategoryTheory.Functor C D
                       A f : C
                       ⊢ ∀ ⦃Y : C⦄ (f_1 : Quiver.Hom f Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                     -/
  naturality f := by simp [prodComparison_natural]
                     /-
                       🎉 no goals
                     -/


@[reassoc]
theorem inv_prodComparison_map_fst [IsIso (prodComparison F A B)] :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                   D : Type u₂
                                                                   inst✝³ : CategoryTheory.Category.{w, u₂} D
                                                                   F : CategoryTheory.Functor C D
                                                                   A B : C
                                                                   inst✝² : CategoryTheory.Limits.HasBinaryProduct A B
                                                                   inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (F.obj A) (F.obj B)
                                                                   inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F A B)
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
                                                                 -/
    inv (prodComparison F A B) ≫ F.map prod.fst = prod.fst := by simp [IsIso.inv_comp_eq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[reassoc]
theorem inv_prodComparison_map_snd [IsIso (prodComparison F A B)] :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                   D : Type u₂
                                                                   inst✝³ : CategoryTheory.Category.{w, u₂} D
                                                                   F : CategoryTheory.Functor C D
                                                                   A B : C
                                                                   inst✝² : CategoryTheory.Limits.HasBinaryProduct A B
                                                                   inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (F.obj A) (F.obj B)
                                                                   inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F A B)
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
                                                                 -/
    inv (prodComparison F A B) ≫ F.map prod.snd = prod.snd := by simp [IsIso.inv_comp_eq]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- If the product comparison morphism is an iso, its inverse is natural. -/
@[reassoc]
theorem prodComparison_inv_natural (f : A ⟶ A') (g : B ⟶ B') [IsIso (prodComparison F A B)]
    [IsIso (prodComparison F A' B')] :
    inv (prodComparison F A B) ≫ F.map (prod.map f g) =
      prod.map (F.map f) (F.map g) ≫ inv (prodComparison F A' B') := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{w, u₂} D
    F : CategoryTheory.Functor C D
    A A' B B' : C
    inst✝⁵ : CategoryTheory.Limits.HasBinaryProduct A B
    inst✝⁴ : CategoryTheory.Limits.HasBinaryProduct A' B'
    inst✝³ : CategoryTheory.Limits.HasBinaryProduct (F.obj A) (F.obj B)
    inst✝² : CategoryTheory.Limits.HasBinaryProduct (F.obj A') (F.obj B')
    f : Quiver.Hom A A'
    g : Quiver.Hom B B'
    inst✝¹ : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F A B)
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison F A' B')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
  -/
  rw [IsIso.eq_comp_inv, Category.assoc, IsIso.inv_comp_eq, prodComparison_natural]
  /-
    🎉 no goals
  -/


/-- The natural isomorphism `F(A ⨯ -) ≅ FA ⨯ F-`, provided each `prodComparison F A B` is an
isomorphism (as `B` changes).
-/
-- @[simps (config := { rhsMd := semireducible })] -- Porting note: no config for semireducible
@[simps]
def prodComparisonNatIso [HasBinaryProducts C] [HasBinaryProducts D] (A : C)
    [∀ B, IsIso (prodComparison F A B)] :
    prod.functor.obj A ⋙ F ≅ F ⋙ prod.functor.obj (F.obj A) := by
  /-
    C : Type u
    inst✝¹¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{w, u₂} D
    E : Type u₃
    inst✝⁹ : CategoryTheory.Category.{w', u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    A✝ A' B B' : C
    inst✝⁸ : CategoryTheory.Limits.HasBinaryProduct A✝ B
    inst✝⁷ : CategoryTheory.Limits.HasBinaryProduct A' B'
    inst✝⁶ : CategoryTheory.Limits.HasBinaryProduct (F.obj A✝) (F.obj B)
    inst✝⁵ : CategoryTheory.Limits.HasBinaryProduct (F.obj A') (F.obj B')
    inst✝⁴ : CategoryTheory.Limits.HasBinaryProduct (G.obj (F.obj A✝)) (G.obj (F.o …
    inst✝³ : CategoryTheory.Limits.HasBinaryProduct ((F.comp G).obj A✝) ((F.comp G …
    inst✝² : CategoryTheory.Limits.HasBinaryProducts C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProducts D
    A : C
    inst✝ : ∀ (B : C), CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison  …
    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.prod.functor.obj A).comp F) (F.co …
  -/
  refine { @asIso _ _ _ _ _ (?_) with hom := prodComparisonNatTrans F A }
  /-
    C : Type u
    inst✝¹¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    D : Type u₂
    inst✝¹⁰ : CategoryTheory.Category.{w, u₂} D
    E : Type u₃
    inst✝⁹ : CategoryTheory.Category.{w', u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    A✝ A' B B' : C
    inst✝⁸ : CategoryTheory.Limits.HasBinaryProduct A✝ B
    inst✝⁷ : CategoryTheory.Limits.HasBinaryProduct A' B'
    inst✝⁶ : CategoryTheory.Limits.HasBinaryProduct (F.obj A✝) (F.obj B)
    inst✝⁵ : CategoryTheory.Limits.HasBinaryProduct (F.obj A') (F.obj B')
    inst✝⁴ : CategoryTheory.Limits.HasBinaryProduct (G.obj (F.obj A✝)) (G.obj (F.o …
    inst✝³ : CategoryTheory.Limits.HasBinaryProduct ((F.comp G).obj A✝) ((F.comp G …
    inst✝² : CategoryTheory.Limits.HasBinaryProducts C
    inst✝¹ : CategoryTheory.Limits.HasBinaryProducts D
    A : C
    inst✝ : ∀ (B : C), CategoryTheory.IsIso (CategoryTheory.Limits.prodComparison  …
    ⊢ CategoryTheory.IsIso { app := fun B => CategoryTheory.Limits.prodComparison  …
  -/
  apply NatIso.isIso_of_isIso_app
  /-
    🎉 no goals
  -/


theorem prodComparison_comp :
    prodComparison (F ⋙ G) A B =
      G.map (prodComparison F A B) ≫ prodComparison G (F.obj A) (F.obj B) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{w, u₂} D
    E : Type u₃
    inst✝⁴ : CategoryTheory.Category.{w', u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    A B : C
    inst✝³ : CategoryTheory.Limits.HasBinaryProduct A B
    inst✝² : CategoryTheory.Limits.HasBinaryProduct (F.obj A) (F.obj B)
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj (F.obj A)) (G.obj (F.ob …
    inst✝ : CategoryTheory.Limits.HasBinaryProduct ((F.comp G).obj A) ((F.comp G). …
    ⊢ Eq (CategoryTheory.Limits.prodComparison (F.comp G) A B) (CategoryTheory.Cat …
  -/
  unfold prodComparison
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{w, u₂} D
    E : Type u₃
    inst✝⁴ : CategoryTheory.Category.{w', u₃} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    A B : C
    inst✝³ : CategoryTheory.Limits.HasBinaryProduct A B
    inst✝² : CategoryTheory.Limits.HasBinaryProduct (F.obj A) (F.obj B)
    inst✝¹ : CategoryTheory.Limits.HasBinaryProduct (G.obj (F.obj A)) (G.obj (F.ob …
    inst✝ : CategoryTheory.Limits.HasBinaryProduct ((F.comp G).obj A) ((F.comp G). …
    ⊢ Eq (CategoryTheory.Limits.prod.lift ((F.comp G).map CategoryTheory.Limits.pr …
  -/
                                         /-
                                           🎉 no goals
                                         -/
  ext <;> simp <;> rw [← G.map_comp] <;> simp
                                         /-
                                           🎉 no goals
                                         -/


/-- The coproduct comparison morphism.

In `CategoryTheory/Limits/Preserves` we show
this is always an iso iff F preserves binary coproducts.
-/
def coprodComparison (F : C ⥤ D) (A B : C) [HasBinaryCoproduct A B]
    [HasBinaryCoproduct (F.obj A) (F.obj B)] : F.obj A ⨿ F.obj B ⟶ F.obj (A ⨿ B) :=
  coprod.desc (F.map coprod.inl) (F.map coprod.inr)


@[reassoc (attr := simp)]
theorem coprodComparison_inl : coprod.inl ≫ coprodComparison F A B = F.map coprod.inl :=
  coprod.inl_desc _ _


@[reassoc (attr := simp)]
theorem coprodComparison_inr : coprod.inr ≫ coprodComparison F A B = F.map coprod.inr :=
  coprod.inr_desc _ _


/-- Naturality of the coprod_comparison morphism in both arguments. -/
@[reassoc]
theorem coprodComparison_natural (f : A ⟶ A') (g : B ⟶ B') :
    coprodComparison F A B ≫ F.map (coprod.map f g) =
      coprod.map (F.map f) (F.map g) ≫ coprodComparison F A' B' := by
  rw [coprodComparison, coprodComparison, coprod.map_desc, ← F.map_comp, ← F.map_comp,
    coprod.desc_comp, ← F.map_comp, coprod.inl_map, ← F.map_comp, coprod.inr_map]


/-- The coproduct comparison morphism from `FA ⨿ F-` to `F(A ⨿ -)`, whose components are given by
`coprodComparison`.
-/
@[simps]
def coprodComparisonNatTrans [HasBinaryCoproducts C] [HasBinaryCoproducts D] (F : C ⥤ D) (A : C) :
    F ⋙ coprod.functor.obj (F.obj A) ⟶ coprod.functor.obj A ⋙ F where
  app B := coprodComparison F A B
                     /-
                       C : Type u
                       inst✝⁷ : CategoryTheory.Category.{v, u} C
                       X Y : C
                       D : Type u₂
                       inst✝⁶ : CategoryTheory.Category.{w, u₂} D
                       F✝ : CategoryTheory.Functor C D
                       A✝ A' B B' : C
                       inst✝⁵ : CategoryTheory.Limits.HasBinaryCoproduct A✝ B
                       inst✝⁴ : CategoryTheory.Limits.HasBinaryCoproduct A' B'
                       inst✝³ : CategoryTheory.Limits.HasBinaryCoproduct (F✝.obj A✝) (F✝.obj B)
                       inst✝² : CategoryTheory.Limits.HasBinaryCoproduct (F✝.obj A') (F✝.obj B')
                       inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts C
                       inst✝ : CategoryTheory.Limits.HasBinaryCoproducts D
                       F : CategoryTheory.Functor C D
                       A f : C
                       ⊢ ∀ ⦃Y : C⦄ (f_1 : Quiver.Hom f Y), Eq (CategoryTheory.CategoryStruct.comp ((F …
                     -/
  naturality f := by simp [coprodComparison_natural]
                     /-
                       🎉 no goals
                     -/


@[reassoc]
theorem map_inl_inv_coprodComparison [IsIso (coprodComparison F A B)] :
                                                                       /-
                                                                         C : Type u
                                                                         inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                         D : Type u₂
                                                                         inst✝³ : CategoryTheory.Category.{w, u₂} D
                                                                         F : CategoryTheory.Functor C D
                                                                         A B : C
                                                                         inst✝² : CategoryTheory.Limits.HasBinaryCoproduct A B
                                                                         inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A) (F.obj B)
                                                                         inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison F A B)
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.coprod.i …
                                                                       -/
    F.map coprod.inl ≫ inv (coprodComparison F A B) = coprod.inl := by simp [IsIso.inv_comp_eq]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[reassoc]
theorem map_inr_inv_coprodComparison [IsIso (coprodComparison F A B)] :
                                                                       /-
                                                                         C : Type u
                                                                         inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                         D : Type u₂
                                                                         inst✝³ : CategoryTheory.Category.{w, u₂} D
                                                                         F : CategoryTheory.Functor C D
                                                                         A B : C
                                                                         inst✝² : CategoryTheory.Limits.HasBinaryCoproduct A B
                                                                         inst✝¹ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A) (F.obj B)
                                                                         inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison F A B)
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map CategoryTheory.Limits.coprod.i …
                                                                       -/
    F.map coprod.inr ≫ inv (coprodComparison F A B) = coprod.inr := by simp [IsIso.inv_comp_eq]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- If the coproduct comparison morphism is an iso, its inverse is natural. -/
@[reassoc]
theorem coprodComparison_inv_natural (f : A ⟶ A') (g : B ⟶ B') [IsIso (coprodComparison F A B)]
    [IsIso (coprodComparison F A' B')] :
    inv (coprodComparison F A B) ≫ coprod.map (F.map f) (F.map g) =
      F.map (coprod.map f g) ≫ inv (coprodComparison F A' B') := by
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    D : Type u₂
    inst✝⁶ : CategoryTheory.Category.{w, u₂} D
    F : CategoryTheory.Functor C D
    A A' B B' : C
    inst✝⁵ : CategoryTheory.Limits.HasBinaryCoproduct A B
    inst✝⁴ : CategoryTheory.Limits.HasBinaryCoproduct A' B'
    inst✝³ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A) (F.obj B)
    inst✝² : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A') (F.obj B')
    f : Quiver.Hom A A'
    g : Quiver.Hom B B'
    inst✝¹ : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison F A B)
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Limits.coprodComparison F A' B')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
  -/
  rw [IsIso.eq_comp_inv, Category.assoc, IsIso.inv_comp_eq, coprodComparison_natural]
  /-
    🎉 no goals
  -/


/-- The natural isomorphism `FA ⨿ F- ≅ F(A ⨿ -)`, provided each `coprodComparison F A B` is an
isomorphism (as `B` changes).
-/
-- @[simps (config := { rhsMd := semireducible })] -- Porting note: no config for semireducible
@[simps]
def coprodComparisonNatIso [HasBinaryCoproducts C] [HasBinaryCoproducts D] (A : C)
    [∀ B, IsIso (coprodComparison F A B)] :
    F ⋙ coprod.functor.obj (F.obj A) ≅ coprod.functor.obj A ⋙ F := by
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    X Y : C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{w, u₂} D
    F : CategoryTheory.Functor C D
    A✝ A' B B' : C
    inst✝⁶ : CategoryTheory.Limits.HasBinaryCoproduct A✝ B
    inst✝⁵ : CategoryTheory.Limits.HasBinaryCoproduct A' B'
    inst✝⁴ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A✝) (F.obj B)
    inst✝³ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A') (F.obj B')
    inst✝² : CategoryTheory.Limits.HasBinaryCoproducts C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts D
    A : C
    inst✝ : ∀ (B : C), CategoryTheory.IsIso (CategoryTheory.Limits.coprodCompariso …
    ⊢ CategoryTheory.Iso (F.comp (CategoryTheory.Limits.coprod.functor.obj (F.obj  …
  -/
  refine { @asIso _ _ _ _ _ (?_) with hom := coprodComparisonNatTrans F A }
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    X Y : C
    D : Type u₂
    inst✝⁷ : CategoryTheory.Category.{w, u₂} D
    F : CategoryTheory.Functor C D
    A✝ A' B B' : C
    inst✝⁶ : CategoryTheory.Limits.HasBinaryCoproduct A✝ B
    inst✝⁵ : CategoryTheory.Limits.HasBinaryCoproduct A' B'
    inst✝⁴ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A✝) (F.obj B)
    inst✝³ : CategoryTheory.Limits.HasBinaryCoproduct (F.obj A') (F.obj B')
    inst✝² : CategoryTheory.Limits.HasBinaryCoproducts C
    inst✝¹ : CategoryTheory.Limits.HasBinaryCoproducts D
    A : C
    inst✝ : ∀ (B : C), CategoryTheory.IsIso (CategoryTheory.Limits.coprodCompariso …
    ⊢ CategoryTheory.IsIso { app := fun B => CategoryTheory.Limits.coprodCompariso …
  -/
  apply NatIso.isIso_of_isIso_app -- Porting note: this did not work inside { }
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `Over.coprod`. -/
@[simps]
noncomputable def Over.coprodObj [HasBinaryCoproducts C] {A : C} :
    Over A → Over A ⥤ Over A :=
  fun f =>
  { obj := fun g => Over.mk (coprod.desc f.hom g.hom)
                    /-
                      C : Type u
                      inst✝¹ : CategoryTheory.Category.{v, u} C
                      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
                      A : C
                      f X✝ Y✝ : CategoryTheory.Over A
                      k : Quiver.Hom X✝ Y✝
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map (Ca …
                    -/
    map := fun k => Over.homMk (coprod.map (𝟙 _) k.left) }
                    /-
                      🎉 no goals
                    -/


/-- A category with binary coproducts has a functorial `sup` operation on over categories. -/
@[simps]
noncomputable def Over.coprod [HasBinaryCoproducts C] {A : C} : Over A ⥤ Over A ⥤ Over A where
  obj f := Over.coprodObj f
  map k :=
    { app := fun g => Over.homMk (coprod.map k.left (𝟙 _)) (by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
          A : C
          X✝ Y✝ : CategoryTheory.Over A
          k : Quiver.Hom X✝ Y✝
          g : CategoryTheory.Over A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coprod.map k.l …
        -/
        dsimp; rw [coprod.map_desc, Category.id_comp, Over.w k])
               /-
                 🎉 no goals
               -/
      naturality := fun f g k => by
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
          A : C
          X✝ Y✝ : CategoryTheory.Over A
          k✝ : Quiver.Hom X✝ Y✝
          f g : CategoryTheory.Over A
          k : Quiver.Hom f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun f => f.coprodObj) X✝).map k) ( …
        -/
        ext
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
          A : C
          X✝ Y✝ : CategoryTheory.Over A
          k✝ : Quiver.Hom X✝ Y✝
          f g : CategoryTheory.Over A
          k : Quiver.Hom f g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun f => f.coprodObj) X✝).map k) ( …
        -/
        dsimp; simp }
               /-
                 🎉 no goals
               -/
  map_id X := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      X : CategoryTheory.Over A
      ⊢ Eq ({ obj := fun f => f.coprodObj, map := fun {X Y} k => { app := fun g => C …
    -/
    ext
    /-
      case w.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      X x✝ : CategoryTheory.Over A
      ⊢ Eq (({ obj := fun f => f.coprodObj, map := fun {X Y} k => { app := fun g =>  …
    -/
    dsimp; simp
           /-
             🎉 no goals
           -/
  map_comp f g := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      X✝ Y✝ Z✝ : CategoryTheory.Over A
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun f => f.coprodObj, map := fun {X Y} k => { app := fun g => C …
    -/
    ext
    /-
      case w.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasBinaryCoproducts C
      A : C
      X✝ Y✝ Z✝ : CategoryTheory.Over A
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      x✝ : CategoryTheory.Over A
      ⊢ Eq (({ obj := fun f => f.coprodObj, map := fun {X Y} k => { app := fun g =>  …
    -/
    dsimp; simp
           /-
             🎉 no goals
           -/


