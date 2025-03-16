/-- A type synonym for `β → C`, used for `β`-graded objects in a category `C`. -/
def GradedObject (β : Type w) (C : Type u) : Type max w u :=
  β → C

-- Satisfying the inhabited linter...

instance inhabitedGradedObject (β : Type w) (C : Type u) [Inhabited C] :
    Inhabited (GradedObject β C) :=
  ⟨fun _ => Inhabited.default⟩

-- `s` is here to distinguish type synonyms asking for different shifts

/-- A type synonym for `β → C`, used for `β`-graded objects in a category `C`
with a shift functor given by translation by `s`.
-/
@[nolint unusedArguments]
abbrev GradedObjectWithShift {β : Type w} [AddCommGroup β] (_ : β) (C : Type u) : Type max w u :=
  GradedObject β C


@[simps!]
instance categoryOfGradedObjects (β : Type w) : Category.{max w v} (GradedObject β C) :=
  CategoryTheory.pi fun _ => C

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/10688): added to ease automation

@[ext]
lemma hom_ext {β : Type*} {X Y : GradedObject β C} (f g : X ⟶ Y) (h : ∀ x, f x = g x) : f = g := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    β : Type u_1
    X Y : CategoryTheory.GradedObject β C
    f g : Quiver.Hom X Y
    h : ∀ (x : β), Eq (f x) (g x)
    ⊢ Eq f g
  -/
  funext
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    β : Type u_1
    X Y : CategoryTheory.GradedObject β C
    f g : Quiver.Hom X Y
    h : ∀ (x : β), Eq (f x) (g x)
    x✝ : β
    ⊢ Eq (f x✝) (g x✝)
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- The projection of a graded object to its `i`-th component. -/
@[simps]
def eval {β : Type w} (b : β) : GradedObject β C ⥤ C where
  obj X := X b
  map f := f b


/-- Constructor for isomorphisms in `GradedObject` -/
@[simps]
def isoMk (e : ∀ i, X i ≅ Y i) : X ≅ Y where
  hom i := (e i).hom
  inv i := (e i).inv


lemma isIso_of_isIso_apply (f : X ⟶ Y) [hf : ∀ i, IsIso (f i)] :
    IsIso f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    β : Type u_1
    X Y : CategoryTheory.GradedObject β C
    f : Quiver.Hom X Y
    hf : ∀ (i : β), CategoryTheory.IsIso (f i)
    ⊢ CategoryTheory.IsIso f
  -/
  change IsIso (isoMk X Y (fun i => asIso (f i))).hom
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    β : Type u_1
    X Y : CategoryTheory.GradedObject β C
    f : Quiver.Hom X Y
    hf : ∀ (i : β), CategoryTheory.IsIso (f i)
    ⊢ CategoryTheory.IsIso (X.isoMk Y fun i => CategoryTheory.asIso (f i)).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance isIso_apply_of_isIso (f : X ⟶ Y) [IsIso f] (i : β) : IsIso (f i) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    β : Type u_1
    X Y : CategoryTheory.GradedObject β C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    i : β
    ⊢ CategoryTheory.IsIso (f i)
  -/
  change IsIso ((eval i).map f)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    β : Type u_1
    X Y : CategoryTheory.GradedObject β C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.IsIso f
    i : β
    ⊢ CategoryTheory.IsIso ((CategoryTheory.GradedObject.eval i).map f)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma hom_inv_id_eval (e : X ≅ Y) (j : J) :
    e.hom j ≫ e.inv j = 𝟙 _ := by
  rw [← GradedObject.categoryOfGradedObjects_comp, e.hom_inv_id,
    GradedObject.categoryOfGradedObjects_id]


@[reassoc (attr := simp)]
lemma inv_hom_id_eval (e : X ≅ Y) (j : J) :
    e.inv j ≫ e.hom j = 𝟙 _ := by
  rw [← GradedObject.categoryOfGradedObjects_comp, e.inv_hom_id,
    GradedObject.categoryOfGradedObjects_id]


@[reassoc (attr := simp)]
lemma map_hom_inv_id_eval (e : X ≅ Y) (F : C ⥤ D) (j : J) :
    F.map (e.hom j) ≫ F.map (e.inv j) = 𝟙 _ := by
  rw [← F.map_comp, ← GradedObject.categoryOfGradedObjects_comp, e.hom_inv_id,
    GradedObject.categoryOfGradedObjects_id, Functor.map_id]


@[reassoc (attr := simp)]
lemma map_inv_hom_id_eval (e : X ≅ Y) (F : C ⥤ D) (j : J) :
    F.map (e.inv j) ≫ F.map (e.hom j) = 𝟙 _ := by
  rw [← F.map_comp, ← GradedObject.categoryOfGradedObjects_comp, e.inv_hom_id,
    GradedObject.categoryOfGradedObjects_id, Functor.map_id]


@[reassoc (attr := simp)]
lemma map_hom_inv_id_eval_app (e : X ≅ Y) (F : C ⥤ D ⥤ E) (j : J) (Y : D) :
    (F.map (e.hom j)).app Y ≫ (F.map (e.inv j)).app Y = 𝟙 _ := by
  rw [← NatTrans.comp_app, ← F.map_comp, hom_inv_id_eval,
    Functor.map_id, NatTrans.id_app]


@[reassoc (attr := simp)]
lemma map_inv_hom_id_eval_app (e : X ≅ Y) (F : C ⥤ D ⥤ E) (j : J) (Y : D) :
    (F.map (e.inv j)).app Y ≫ (F.map (e.hom j)).app Y = 𝟙 _ := by
  rw [← NatTrans.comp_app, ← F.map_comp, inv_hom_id_eval,
    Functor.map_id, NatTrans.id_app]


/-- Pull back an `I`-graded object in `C` to a `J`-graded object along a function `J → I`. -/
abbrev comap {I J : Type*} (h : J → I) : GradedObject I C ⥤ GradedObject J C :=
  Pi.comap (fun _ => C) h

-- Porting note: added to ease the port, this is a special case of `Functor.eqToHom_proj`

@[simp]
theorem eqToHom_proj {I : Type*} {x x' : GradedObject I C} (h : x = x') (i : I) :
    (eqToHom h : x ⟶ x') i = eqToHom (funext_iff.mp h i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I : Type u_1
    x x' : CategoryTheory.GradedObject I C
    h : Eq x x'
    i : I
    ⊢ Eq (CategoryTheory.eqToHom h i) (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I : Type u_1
    x : CategoryTheory.GradedObject I C
    i : I
    ⊢ Eq (CategoryTheory.eqToHom ⋯ i) (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The natural isomorphism comparing between
pulling back along two propositionally equal functions.
-/
@[simps]
def comapEq {β γ : Type w} {f g : β → γ} (h : f = g) : comap C f ≅ comap C g where
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           β γ : Type w
                                           f g : β → γ
                                           h : Eq f g
                                           X : CategoryTheory.GradedObject γ C
                                           b : β
                                           ⊢ Eq ((CategoryTheory.GradedObject.comap C f).obj X b) ((CategoryTheory.Graded …
                                         -/
  hom := { app := fun X b => eqToHom (by dsimp; simp only [h]) }
                                                /-
                                                  🎉 no goals
                                                -/
                                         /-
                                           C : Type u
                                           inst✝ : CategoryTheory.Category.{v, u} C
                                           β γ : Type w
                                           f g : β → γ
                                           h : Eq f g
                                           X : CategoryTheory.GradedObject γ C
                                           b : β
                                           ⊢ Eq ((CategoryTheory.GradedObject.comap C g).obj X b) ((CategoryTheory.Graded …
                                         -/
  inv := { app := fun X b => eqToHom (by dsimp; simp only [h]) }
                                                /-
                                                  🎉 no goals
                                                -/


theorem comapEq_symm {β γ : Type w} {f g : β → γ} (h : f = g) :
                                                /-
                                                  C : Type u
                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                  β γ : Type w
                                                  f g : β → γ
                                                  h : Eq f g
                                                  ⊢ Eq (CategoryTheory.GradedObject.comapEq C ⋯) (CategoryTheory.GradedObject.co …
                                                -/
    comapEq C h.symm = (comapEq C h).symm := by aesop_cat
                                                /-
                                                  🎉 no goals
                                                -/


theorem comapEq_trans {β γ : Type w} {f g h : β → γ} (k : f = g) (l : g = h) :
                                                             /-
                                                               C : Type u
                                                               inst✝ : CategoryTheory.Category.{v, u} C
                                                               β γ : Type w
                                                               f g h : β → γ
                                                               k : Eq f g
                                                               l : Eq g h
                                                               ⊢ Eq (CategoryTheory.GradedObject.comapEq C ⋯) ((CategoryTheory.GradedObject.c …
                                                             -/
    comapEq C (k.trans l) = comapEq C k ≪≫ comapEq C l := by aesop_cat
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem eqToHom_apply {β : Type w} {X Y : β → C} (h : X = Y) (b : β) :
                                        /-
                                          C : Type u
                                          inst✝ : CategoryTheory.Category.{v, u} C
                                          β : Type w
                                          X Y : β → C
                                          h : Eq X Y
                                          b : β
                                          ⊢ Eq (X b) (Y b)
                                        -/
    (eqToHom h : X ⟶ Y) b = eqToHom (by rw [h]) := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    β : Type w
    X Y : β → C
    h : Eq X Y
    b : β
    ⊢ Eq (CategoryTheory.eqToHom h b) (CategoryTheory.eqToHom ⋯)
  -/
  subst h
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    β : Type w
    X : β → C
    b : β
    ⊢ Eq (CategoryTheory.eqToHom ⋯ b) (CategoryTheory.eqToHom ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The equivalence between β-graded objects and γ-graded objects,
given an equivalence between β and γ.
-/
@[simps]
def comapEquiv {β γ : Type w} (e : β ≃ γ) : GradedObject β C ≌ GradedObject γ C where
  functor := comap C (e.symm : γ → β)
  inverse := comap C (e : β → γ)
  counitIso :=
                                                         /-
                                                           C : Type u
                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                           β γ : Type w
                                                           e : Equiv β γ
                                                           ⊢ Eq (Function.comp ⇑e ⇑e.symm) fun i => i
                                                         -/
    (Pi.comapComp (fun _ => C) _ _).trans (comapEq C (by ext; simp))
                   /-
                     C : Type u
                     inst✝ : CategoryTheory.Category.{v, u} C
                     β γ : Type w
                     e : Equiv β γ
                     ⊢ Eq (fun i => i) (Function.comp ⇑e.symm ⇑e)
                   -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                        /-
                          🎉 no goals
                        -/
  unitIso :=
    (comapEq C (by ext; simp)).trans (Pi.comapComp _ _ _).symm


instance hasShift {β : Type*} [AddCommGroup β] (s : β) : HasShift (GradedObjectWithShift s C) ℤ :=
  hasShiftMk _ _
    { F := fun n => comap C fun b : β => b + n • s
                            /-
                              C : Type u
                              inst✝¹ : CategoryTheory.Category.{v, u} C
                              β : Type u_1
                              inst✝ : AddCommGroup β
                              s : β
                              ⊢ Eq (fun b => HAdd.hAdd b (HSMul.hSMul 0 s)) id
                            -/
      zero := comapEq C (by aesop_cat) ≪≫ Pi.comapId β fun _ => C
                            /-
                              🎉 no goals
                            -/
                                      /-
                                        C : Type u
                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                        β : Type u_1
                                        inst✝ : AddCommGroup β
                                        s : β
                                        m n : Int
                                        ⊢ Eq (fun b => HAdd.hAdd b (HSMul.hSMul (HAdd.hAdd m n) s)) (Function.comp (fu …
                                      -/
      add := fun m n => comapEq C (by ext; dsimp; rw [add_comm m n, add_zsmul, add_assoc]) ≪≫
                                                  /-
                                                    🎉 no goals
                                                  -/
          (Pi.comapComp _ _ _).symm }


@[simp]
theorem shiftFunctor_obj_apply {β : Type*} [AddCommGroup β] (s : β) (X : β → C) (t : β) (n : ℤ) :
    (shiftFunctor (GradedObjectWithShift s C) n).obj X t = X (t + n • s) :=
  rfl


@[simp]
theorem shiftFunctor_map_apply {β : Type*} [AddCommGroup β] (s : β)
    {X Y : GradedObjectWithShift s C} (f : X ⟶ Y) (t : β) (n : ℤ) :
    (shiftFunctor (GradedObjectWithShift s C) n).map f t = f (t + n • s) :=
  rfl


instance [HasZeroMorphisms C] (β : Type w) (X Y : GradedObject β C) :
  Zero (X ⟶ Y) := ⟨fun _ => 0⟩


@[simp, nolint simpNF]
theorem zero_apply [HasZeroMorphisms C] (β : Type w) (X Y : GradedObject β C) (b : β) :
    (0 : X ⟶ Y) b = 0 :=
  rfl


instance hasZeroMorphisms [HasZeroMorphisms C] (β : Type w) :
    HasZeroMorphisms.{max w v} (GradedObject β C) where


instance hasZeroObject [HasZeroObject C] [HasZeroMorphisms C] (β : Type w) :
    HasZeroObject.{max w v} (GradedObject β C) := by
  refine ⟨⟨fun _ => 0, fun X => ⟨⟨⟨fun b => 0⟩, fun f => ?_⟩⟩, fun X =>
                                        /-
                                          case refine_1
                                          C : Type u
                                          inst✝² : CategoryTheory.Category.{v, u} C
                                          inst✝¹ : CategoryTheory.Limits.HasZeroObject C
                                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                          β : Type w
                                          X : CategoryTheory.GradedObject β C
                                          f : Quiver.Hom (fun x => 0) X
                                          ⊢ Eq f Inhabited.default
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
    ⟨⟨⟨fun b => 0⟩, fun f => ?_⟩⟩⟩⟩ <;> aesop_cat
                                        /-
                                          🎉 no goals
                                        -/


/-- The total object of a graded object is the coproduct of the graded components.
-/
noncomputable def total : GradedObject β C ⥤ C where
  obj X := ∐ fun i : β => X i
  map f := Limits.Sigma.map fun i => f i


/--
The `total` functor taking a graded object to the coproduct of its graded components is faithful.
To prove this, we need to know that the coprojections into the coproduct are monomorphisms,
which follows from the fact we have zero morphisms and decidable equality for the grading.
-/
instance : (total β C).Faithful where
  map_injective {X Y} f g w := by
    /-
      β : Type
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : CategoryTheory.GradedObject β C
      f g : Quiver.Hom X Y
      w : Eq ((CategoryTheory.GradedObject.total β C).map f) ((CategoryTheory.Graded …
      ⊢ Eq f g
    -/
    ext i
    /-
      case h
      β : Type
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : CategoryTheory.GradedObject β C
      f g : Quiver.Hom X Y
      w : Eq ((CategoryTheory.GradedObject.total β C).map f) ((CategoryTheory.Graded …
      i : β
      ⊢ Eq (f i) (g i)
    -/
    replace w := Sigma.ι (fun i : β => X i) i ≫= w
    /-
      case h
      β : Type
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : CategoryTheory.GradedObject β C
      f g : Quiver.Hom X Y
      i : β
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun …
      ⊢ Eq (f i) (g i)
    -/
    erw [colimit.ι_map, colimit.ι_map] at w
    /-
      case h
      β : Type
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : CategoryTheory.GradedObject β C
      f g : Quiver.Hom X Y
      i : β
      w : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Discrete.natTrans  …
      ⊢ Eq (f i) (g i)
    -/
    simp? at * says simp only [Discrete.functor_obj_eq_as, Discrete.natTrans_app] at *
    /-
      case h
      β : Type
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Limits.HasCoproducts C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : CategoryTheory.GradedObject β C
      f g : Quiver.Hom X Y
      i : β
      w : Eq (CategoryTheory.CategoryStruct.comp (f i) (CategoryTheory.Limits.colimi …
      ⊢ Eq (f i) (g i)
    -/
    exact Mono.right_cancellation _ _ w
    /-
      🎉 no goals
    -/


instance : ConcreteCategory (GradedObject β C) where forget := total β C ⋙ forget C


instance : HasForget₂ (GradedObject β C) C where forget₂ := total β C


/-- If `X : GradedObject I C` and `p : I → J`, `X.mapObjFun p j` is the family of objects `X i`
for `i : I` such that `p i = j`. -/
abbrev mapObjFun (j : J) (i : p ⁻¹' {j}) : C := X i


/-- Given `X : GradedObject I C` and `p : I → J`, `X.HasMap p` is the condition that
for all `j : J`, the coproduct of all `X i` such `p i = j` exists. -/
abbrev HasMap : Prop := ∀ (j : J), HasCoproduct (X.mapObjFun p j)


variable {X Y} in
lemma hasMap_of_iso (e : X ≅ Y) (p: I → J) [HasMap X p] : HasMap Y p := fun j => by
  have α : Discrete.functor (X.mapObjFun p j) ≅ Discrete.functor (Y.mapObjFun p j) :=
    Discrete.natIso (fun ⟨i, _⟩ => (GradedObject.eval i).mapIso e)
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} C
    X Y : CategoryTheory.GradedObject I C
    e : CategoryTheory.Iso X Y
    p : I → J
    inst✝ : X.HasMap p
    j : J
    α : CategoryTheory.Iso (CategoryTheory.Discrete.functor (X.mapObjFun p j)) (Ca …
    ⊢ CategoryTheory.Limits.HasCoproduct (Y.mapObjFun p j)
  -/
  exact hasColimitOfIso α.symm
  /-
    🎉 no goals
  -/


/-- Given `X : GradedObject I C` and `p : I → J`, `X.mapObj p` is the graded object by `J`
which in degree `j` consists of the coproduct of the `X i` such that `p i = j`. -/
noncomputable def mapObj : GradedObject J C := fun j => ∐ (X.mapObjFun p j)


/-- The canonical inclusion `X i ⟶ X.mapObj p j` when `i : I` and `j : J` are such
that `p i = j`. -/
noncomputable def ιMapObj (i : I) (j : J) (hij : p i = j) : X i ⟶ X.mapObj p j :=
  Sigma.ι (X.mapObjFun p j) ⟨i, hij⟩


/-- Given `X : GradedObject I C`, `p : I → J` and `j : J`,
`CofanMapObjFun X p j` is the type `Cofan (X.mapObjFun p j)`. The point object of
such colimits cofans are isomorphic to `X.mapObj p j`, see `CofanMapObjFun.iso`. -/
abbrev CofanMapObjFun (j : J) : Type _ := Cofan (X.mapObjFun p j)

-- in order to use the cofan API, some definitions below
-- have a `simp` attribute rather than `simps`

/-- Constructor for `CofanMapObjFun X p j`. -/
@[simp]
def CofanMapObjFun.mk (j : J) (pt : C) (ι' : ∀ (i : I) (_ : p i = j), X i ⟶ pt) :
    CofanMapObjFun X p j :=
  Cofan.mk pt (fun ⟨i, hi⟩ => ι' i hi)


/-- The tautological cofan corresponding to the coproduct decomposition of `X.mapObj p j`. -/
@[simp]
noncomputable def cofanMapObj (j : J) : CofanMapObjFun X p j :=
  CofanMapObjFun.mk X p j (X.mapObj p j) (fun i hi => X.ιMapObj p i j hi)


/-- Given `X : GradedObject I C`, `p : I → J` and `j : J`, `X.mapObj p j` satisfies
the universal property of the coproduct of those `X i` such that `p i = j`. -/
noncomputable def isColimitCofanMapObj (j : J) : IsColimit (X.cofanMapObj p j) :=
  colimit.isColimit _


@[ext]
lemma mapObj_ext {A : C} {j : J} (f g : X.mapObj p j ⟶ A)
    (hfg : ∀ (i : I) (hij : p i = j), X.ιMapObj p i j hij ≫ f = X.ιMapObj p i j hij ≫ g) :
    f = g :=
  Cofan.IsColimit.hom_ext (X.isColimitCofanMapObj p j) _ _ (fun ⟨i, hij⟩ => hfg i hij)


/-- This is the morphism `X.mapObj p j ⟶ A` constructed from a family of
morphisms `X i ⟶ A` for all `i : I` such that `p i = j`. -/
noncomputable def descMapObj {A : C} {j : J} (φ : ∀ (i : I) (_ : p i = j), X i ⟶ A) :
    X.mapObj p j ⟶ A :=
  Cofan.IsColimit.desc (X.isColimitCofanMapObj p j) (fun ⟨i, hi⟩ => φ i hi)


@[reassoc (attr := simp)]
lemma ι_descMapObj {A : C} {j : J}
    (φ : ∀ (i : I) (_ : p i = j), X i ⟶ A) (i : I) (hi : p i = j) :
    X.ιMapObj p i j hi ≫ X.descMapObj p φ = φ i hi := by
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} C
    X : CategoryTheory.GradedObject I C
    p : I → J
    inst✝ : X.HasMap p
    A : C
    j : J
    φ : (i : I) → Eq (p i) j → Quiver.Hom (X i) A
    i : I
    hi : Eq (p i) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ιMapObj p i j hi) (X.descMapObj p  …
  -/
  apply Cofan.IsColimit.fac
  /-
    🎉 no goals
  -/


lemma hasMap (c : ∀ j, CofanMapObjFun X p j) (hc : ∀ j, IsColimit (c j)) :
    X.HasMap p := fun j => ⟨_, hc j⟩


/-- If `c : CofanMapObjFun X p j` is a colimit cofan, this is the induced
isomorphism `c.pt ≅ X.mapObj p j`. -/
noncomputable def iso : c.pt ≅ X.mapObj p j :=
  IsColimit.coconePointUniqueUpToIso hc (X.isColimitCofanMapObj p j)


@[reassoc (attr := simp)]
lemma inj_iso_hom (i : I) (hi : p i = j) :
    c.inj ⟨i, hi⟩ ≫ (c.iso hc).hom = X.ιMapObj p i j hi := by
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} C
    X : CategoryTheory.GradedObject I C
    p : I → J
    j : J
    inst✝ : X.HasMap p
    c : X.CofanMapObjFun p j
    hc : CategoryTheory.Limits.IsColimit c
    i : I
    hi : Eq (p i) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj c ⟨i …
  -/
  apply IsColimit.comp_coconePointUniqueUpToIso_hom
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιMapObj_iso_inv (i : I) (hi : p i = j) :
    X.ιMapObj p i j hi ≫ (c.iso hc).inv = c.inj ⟨i, hi⟩ := by
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝¹ : CategoryTheory.Category.{u_5, u_4} C
    X : CategoryTheory.GradedObject I C
    p : I → J
    j : J
    inst✝ : X.HasMap p
    c : X.CofanMapObjFun p j
    hc : CategoryTheory.Limits.IsColimit c
    i : I
    hi : Eq (p i) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ιMapObj p i j hi) (CategoryTheory. …
  -/
  apply IsColimit.comp_coconePointUniqueUpToIso_inv
  /-
    🎉 no goals
  -/


/-- The canonical morphism of `J`-graded objects `X.mapObj p ⟶ Y.mapObj p` induced by
a morphism `X ⟶ Y` of `I`-graded objects and a map `p : I → J`. -/
noncomputable def mapMap : X.mapObj p ⟶ Y.mapObj p := fun j =>
  X.descMapObj p (fun i hi => φ i ≫ Y.ιMapObj p i j hi)


@[reassoc (attr := simp)]
lemma ι_mapMap (i : I) (j : J) (hij : p i = j) :
    X.ιMapObj p i j hij ≫ mapMap φ p j = φ i ≫ Y.ιMapObj p i j hij := by
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} C
    X Y : CategoryTheory.GradedObject I C
    φ : Quiver.Hom X Y
    p : I → J
    inst✝¹ : X.HasMap p
    inst✝ : Y.HasMap p
    i : I
    j : J
    hij : Eq (p i) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ιMapObj p i j hij) (CategoryTheory …
  -/
  simp only [mapMap, ι_descMapObj]
  /-
    🎉 no goals
  -/


lemma congr_mapMap (φ₁ φ₂ : X ⟶ Y) (h : φ₁ = φ₂) : mapMap φ₁ p = mapMap φ₂ p := by
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} C
    X Y : CategoryTheory.GradedObject I C
    p : I → J
    inst✝¹ : X.HasMap p
    inst✝ : Y.HasMap p
    φ₁ φ₂ : Quiver.Hom X Y
    h : Eq φ₁ φ₂
    ⊢ Eq (CategoryTheory.GradedObject.mapMap φ₁ p) (CategoryTheory.GradedObject.ma …
  -/
  subst h
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} C
    X Y : CategoryTheory.GradedObject I C
    p : I → J
    inst✝¹ : X.HasMap p
    inst✝ : Y.HasMap p
    φ₁ : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.GradedObject.mapMap φ₁ p) (CategoryTheory.GradedObject.ma …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
                                             /-
                                               I : Type u_1
                                               J : Type u_2
                                               C : Type u_4
                                               inst✝¹ : CategoryTheory.Category.{u_5, u_4} C
                                               X : CategoryTheory.GradedObject I C
                                               p : I → J
                                               inst✝ : X.HasMap p
                                               ⊢ Eq (CategoryTheory.GradedObject.mapMap (CategoryTheory.CategoryStruct.id X)  …
                                             -/
lemma mapMap_id : mapMap (𝟙 X) p = 𝟙 _ := by aesop_cat
                                             /-
                                               🎉 no goals
                                             -/


@[simp, reassoc]
                                                                                  /-
                                                                                    I : Type u_1
                                                                                    J : Type u_2
                                                                                    C : Type u_4
                                                                                    inst✝³ : CategoryTheory.Category.{u_5, u_4} C
                                                                                    X Y Z : CategoryTheory.GradedObject I C
                                                                                    φ : Quiver.Hom X Y
                                                                                    ψ : Quiver.Hom Y Z
                                                                                    p : I → J
                                                                                    inst✝² : X.HasMap p
                                                                                    inst✝¹ : Y.HasMap p
                                                                                    inst✝ : Z.HasMap p
                                                                                    ⊢ Eq (CategoryTheory.GradedObject.mapMap (CategoryTheory.CategoryStruct.comp φ …
                                                                                  -/
lemma mapMap_comp [Z.HasMap p] : mapMap (φ ≫ ψ) p = mapMap φ p ≫ mapMap ψ p := by aesop_cat
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The isomorphism of `J`-graded objects `X.mapObj p ≅ Y.mapObj p` induced by an
isomorphism `X ≅ Y` of graded objects and a map `p : I → J`. -/
@[simps]
noncomputable def mapIso : X.mapObj p ≅ Y.mapObj p where
  hom := mapMap e.hom p
  inv := mapMap e.inv p


/-- Given a map `p : I → J`, this is the functor `GradedObject I C ⥤ GradedObject J C` which
sends an `I`-object `X` to the graded object `X.mapObj p` which in degree `j : J` is given
by the coproduct of those `X i` such that `p i = j`. -/
@[simps]
noncomputable def map [∀ (j : J), HasColimitsOfShape (Discrete (p ⁻¹' {j})) C] :
    GradedObject I C ⥤ GradedObject J C where
  obj X := X.mapObj p
  map φ := mapMap φ p


/-- Given maps `p : I → J`, `q : J → K` and `r : I → K` such that `q.comp p = r`,
`X : GradedObject I C`, `k : K`, the datum of cofans `X.CofanMapObjFun p j` for all
`j : J` and of a cofan for all the points of these cofans, this is a cofan of
type `X.CofanMapObjFun r k`, which is a colimit (see `isColimitCofanMapObjComp`) if the
given cofans are. -/
@[simp]
def cofanMapObjComp : X.CofanMapObjFun r k :=
  CofanMapObjFun.mk _ _ _ c'.pt (fun i hi =>
                 /-
                   I : Type u_1
                   J : Type u_2
                   K : Type u_3
                   C : Type u_4
                   inst✝² : CategoryTheory.Category.{?u.143412, u_4} C
                   X Y Z : CategoryTheory.GradedObject I C
                   φ : Quiver.Hom X Y
                   e : CategoryTheory.Iso X Y
                   ψ : Quiver.Hom Y Z
                   p : I → J
                   j : J
                   inst✝¹ : X.HasMap p
                   inst✝ : Y.HasMap p
                   q : J → K
                   r : I → K
                   hpqr : ∀ (i : I), Eq (q (p i)) (r i)
                   k : K
                   c : (j : J) → Eq (q j) k → X.CofanMapObjFun p j
                   hc : (j : J) → (hj : Eq (q j) k) → CategoryTheory.Limits.IsColimit (c j hj)
                   c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
                   hc' : CategoryTheory.Limits.IsColimit c'
                   i : I
                   hi : Eq (r i) k
                   ⊢ Eq (q (p i)) k
                 -/
    (c (p i) (by rw [hpqr, hi])).inj ⟨i, rfl⟩ ≫ c'.inj (⟨p i, by
                 /-
                   🎉 no goals
                 -/
      /-
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.143412, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        k : K
        c : (j : J) → Eq (q j) k → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) k) → CategoryTheory.Limits.IsColimit (c j hj)
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        i : I
        hi : Eq (r i) k
        ⊢ Membership.mem (Set.preimage q (Singleton.singleton k)) (p i)
      -/
      rw [Set.mem_preimage, Set.mem_singleton_iff, hpqr, hi]⟩))
      /-
        🎉 no goals
      -/


/-- Given maps `p : I → J`, `q : J → K` and `r : I → K` such that `q.comp p = r`,
`X : GradedObject I C`, `k : K`, the cofan constructed by `cofanMapObjComp` is a colimit.
In other words, if we have, for all `j : J` such that `hj : q j = k`,
a colimit cofan `c j hj` which computes the coproduct of the `X i` such that `p i = j`,
and also a colimit cofan which computes the coproduct of the points of these `c j hj`, then
the point of this latter cofan computes the coproduct of the `X i` such that `r i = k`. -/
@[simp]
def isColimitCofanMapObjComp :
    IsColimit (cofanMapObjComp X p q r hpqr k c c') :=
  mkCofanColimit _
    (fun s => Cofan.IsColimit.desc hc'
      (fun ⟨j, (hj : q j = k)⟩ => Cofan.IsColimit.desc (hc j hj)
        (fun ⟨i, (hi : p i = j)⟩ => s.inj ⟨i, by
          /-
            I : Type u_1
            J : Type u_2
            K : Type u_3
            C : Type u_4
            inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
            X Y Z : CategoryTheory.GradedObject I C
            φ : Quiver.Hom X Y
            e : CategoryTheory.Iso X Y
            ψ : Quiver.Hom Y Z
            p : I → J
            j✝ : J
            inst✝¹ : X.HasMap p
            inst✝ : Y.HasMap p
            q : J → K
            r : I → K
            hpqr : ∀ (i : I), Eq (q (p i)) (r i)
            k : K
            c : (j : J) → Eq (q j) k → X.CofanMapObjFun p j
            hc : (j : J) → (hj : Eq (q j) k) → CategoryTheory.Limits.IsColimit (c j hj)
            c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
            hc' : CategoryTheory.Limits.IsColimit c'
            s : CategoryTheory.Limits.Cofan (X.mapObjFun r k)
            x✝¹ : ↑(Set.preimage q (Singleton.singleton k))
            j : J
            hj : Eq (q j) k
            x✝ : ↑(Set.preimage p (Singleton.singleton j))
            i : I
            hi : Eq (p i) j
            ⊢ Membership.mem (Set.preimage r (Singleton.singleton k)) i
          -/
          simp only [Set.mem_preimage, Set.mem_singleton_iff, ← hpqr, hi, hj]⟩)))
          /-
            🎉 no goals
          -/
                                     /-
                                       I : Type u_1
                                       J : Type u_2
                                       K : Type u_3
                                       C : Type u_4
                                       inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
                                       X Y Z : CategoryTheory.GradedObject I C
                                       φ : Quiver.Hom X Y
                                       e : CategoryTheory.Iso X Y
                                       ψ : Quiver.Hom Y Z
                                       p : I → J
                                       j : J
                                       inst✝¹ : X.HasMap p
                                       inst✝ : Y.HasMap p
                                       q : J → K
                                       r : I → K
                                       hpqr : ∀ (i : I), Eq (q (p i)) (r i)
                                       k : K
                                       c : (j : J) → Eq (q j) k → X.CofanMapObjFun p j
                                       hc : (j : J) → (hj : Eq (q j) k) → CategoryTheory.Limits.IsColimit (c j hj)
                                       c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
                                       hc' : CategoryTheory.Limits.IsColimit c'
                                       s : CategoryTheory.Limits.Cofan (X.mapObjFun r k)
                                       x✝ : ↑(Set.preimage r (Singleton.singleton k))
                                       i : I
                                       hi : Eq (r i) k
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (X.c …
                                     -/
    (fun s ⟨i, (hi : r i = k)⟩ => by simp)
                                     /-
                                       🎉 no goals
                                     -/
    (fun s m hm => by
      /-
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        k : K
        c : (j : J) → Eq (q j) k → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) k) → CategoryTheory.Limits.IsColimit (c j hj)
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r k)
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr k c c').pt s.pt
        hm : ∀ (j : ↑(Set.preimage r (Singleton.singleton k))), Eq (CategoryTheory.Cat …
        ⊢ Eq m ((fun s => CategoryTheory.Limits.Cofan.IsColimit.desc hc' fun x => Cate …
      -/
      apply Cofan.IsColimit.hom_ext hc'
      /-
        case h
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        k : K
        c : (j : J) → Eq (q j) k → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) k) → CategoryTheory.Limits.IsColimit (c j hj)
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r k)
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr k c c').pt s.pt
        hm : ∀ (j : ↑(Set.preimage r (Singleton.singleton k))), Eq (CategoryTheory.Cat …
        ⊢ ∀ (i : ↑(Set.preimage q (Singleton.singleton k))), Eq (CategoryTheory.Catego …
      -/
      rintro ⟨j, rfl : q j = k⟩
      /-
        case h.mk
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j✝ : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        j : J
        c : (j_1 : J) → Eq (q j_1) (q j) → X.CofanMapObjFun p j_1
        hc : (j_1 : J) → (hj : Eq (q j_1) (q j)) → CategoryTheory.Limits.IsColimit (c  …
        c' : CategoryTheory.Limits.Cofan fun j_1 => (c ↑j_1 ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r (q j))
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr (q j) c c').pt s.pt
        hm : ∀ (j_1 : ↑(Set.preimage r (Singleton.singleton (q j)))), Eq (CategoryTheo …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (c'.inj ⟨j, ⋯⟩) m) (CategoryTheory.Ca …
      -/
      apply Cofan.IsColimit.hom_ext (hc j rfl)
      /-
        case h.mk.h
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j✝ : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        j : J
        c : (j_1 : J) → Eq (q j_1) (q j) → X.CofanMapObjFun p j_1
        hc : (j_1 : J) → (hj : Eq (q j_1) (q j)) → CategoryTheory.Limits.IsColimit (c  …
        c' : CategoryTheory.Limits.Cofan fun j_1 => (c ↑j_1 ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r (q j))
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr (q j) c c').pt s.pt
        hm : ∀ (j_1 : ↑(Set.preimage r (Singleton.singleton (q j)))), Eq (CategoryTheo …
        ⊢ ∀ (i : ↑(Set.preimage p (Singleton.singleton j))), Eq (CategoryTheory.Catego …
      -/
      rintro ⟨i, rfl : p i = j⟩
      /-
        case h.mk.h.mk
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        i : I
        c : (j : J) → Eq (q j) (q (p i)) → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) (q (p i))) → CategoryTheory.Limits.IsColimit (c  …
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r (q (p i)))
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr (q (p i)) c c').pt s.pt
        hm : ∀ (j : ↑(Set.preimage r (Singleton.singleton (q (p i))))), Eq (CategoryTh …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (c ( …
      -/
      dsimp
      /-
        case h.mk.h.mk
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        i : I
        c : (j : J) → Eq (q j) (q (p i)) → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) (q (p i))) → CategoryTheory.Limits.IsColimit (c  …
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r (q (p i)))
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr (q (p i)) c c').pt s.pt
        hm : ∀ (j : ↑(Set.preimage r (Singleton.singleton (q (p i))))), Eq (CategoryTh …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (c ( …
      -/
      rw [Cofan.IsColimit.fac, Cofan.IsColimit.fac, ← hm]
      /-
        case h.mk.h.mk
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        i : I
        c : (j : J) → Eq (q j) (q (p i)) → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) (q (p i))) → CategoryTheory.Limits.IsColimit (c  …
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r (q (p i)))
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr (q (p i)) c c').pt s.pt
        hm : ∀ (j : ↑(Set.preimage r (Singleton.singleton (q (p i))))), Eq (CategoryTh …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (c ( …
      -/
      dsimp
      /-
        case h.mk.h.mk
        I : Type u_1
        J : Type u_2
        K : Type u_3
        C : Type u_4
        inst✝² : CategoryTheory.Category.{?u.145264, u_4} C
        X Y Z : CategoryTheory.GradedObject I C
        φ : Quiver.Hom X Y
        e : CategoryTheory.Iso X Y
        ψ : Quiver.Hom Y Z
        p : I → J
        j : J
        inst✝¹ : X.HasMap p
        inst✝ : Y.HasMap p
        q : J → K
        r : I → K
        hpqr : ∀ (i : I), Eq (q (p i)) (r i)
        i : I
        c : (j : J) → Eq (q j) (q (p i)) → X.CofanMapObjFun p j
        hc : (j : J) → (hj : Eq (q j) (q (p i))) → CategoryTheory.Limits.IsColimit (c  …
        c' : CategoryTheory.Limits.Cofan fun j => (c ↑j ⋯).pt
        hc' : CategoryTheory.Limits.IsColimit c'
        s : CategoryTheory.Limits.Cofan (X.mapObjFun r (q (p i)))
        m : Quiver.Hom (X.cofanMapObjComp p q r hpqr (q (p i)) c c').pt s.pt
        hm : ∀ (j : ↑(Set.preimage r (Singleton.singleton (q (p i))))), Eq (CategoryTh …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofan.inj (c ( …
      -/
      rw [assoc])
      /-
        🎉 no goals
      -/


include hpqr in
lemma hasMap_comp [(X.mapObj p).HasMap q] : X.HasMap r :=
  fun k => ⟨_, isColimitCofanMapObjComp X p q r hpqr k _
    (fun j _ => X.isColimitCofanMapObj p j) _ ((X.mapObj p).isColimitCofanMapObj q k)⟩


/-- The canonical inclusion `X i ⟶ X.mapObj p j` when `p i = j`, the zero morphism otherwise. -/
noncomputable def ιMapObjOrZero : X i ⟶ X.mapObj p j :=
  if h : p i = j
    then X.ιMapObj p i j h
    else 0


lemma ιMapObjOrZero_eq (h : p i = j) : X.ιMapObjOrZero p i j = X.ιMapObj p i j h := dif_pos h


lemma ιMapObjOrZero_eq_zero (h : p i ≠ j) : X.ιMapObjOrZero p i j = 0 := dif_neg h


variable {X Y} in
@[reassoc (attr := simp)]
lemma ιMapObjOrZero_mapMap :
    X.ιMapObjOrZero p i j ≫ mapMap φ p j = φ i ≫ Y.ιMapObjOrZero p i j := by
  /-
    I : Type u_1
    J : Type u_2
    C : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_5, u_4} C
    X Y : CategoryTheory.GradedObject I C
    φ : Quiver.Hom X Y
    p : I → J
    inst✝³ : X.HasMap p
    inst✝² : Y.HasMap p
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : DecidableEq J
    i : I
    j : J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ιMapObjOrZero p i j) (CategoryTheo …
  -/
  by_cases h : p i = j
    /-
      case pos
      I : Type u_1
      J : Type u_2
      C : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_4} C
      X Y : CategoryTheory.GradedObject I C
      φ : Quiver.Hom X Y
      p : I → J
      inst✝³ : X.HasMap p
      inst✝² : Y.HasMap p
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : DecidableEq J
      i : I
      j : J
      h : Eq (p i) j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ιMapObjOrZero p i j) (CategoryTheo …
    -/
  · simp only [ιMapObjOrZero_eq _ _ _ _ h, ι_mapMap]
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u_1
      J : Type u_2
      C : Type u_4
      inst✝⁴ : CategoryTheory.Category.{u_5, u_4} C
      X Y : CategoryTheory.GradedObject I C
      φ : Quiver.Hom X Y
      p : I → J
      inst✝³ : X.HasMap p
      inst✝² : Y.HasMap p
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝ : DecidableEq J
      i : I
      j : J
      h : Not (Eq (p i) j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.ιMapObjOrZero p i j) (CategoryTheo …
    -/
  · simp only [ιMapObjOrZero_eq_zero _ _ _ _ h, zero_comp, comp_zero]
    /-
      🎉 no goals
    -/


