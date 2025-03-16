local macro:1000 (priority := high) X:term " _[" n:term "]₂" : term =>
    `(($X : SSet.Truncated 2).obj (Opposite.op ⟨SimplexCategory.mk $n, by decide⟩))


local macro:max (priority := high) "[" n:term "]₂" : term =>
  `((⟨SimplexCategory.mk $n, by decide⟩ : SimplexCategory.Truncated 2))


/-- A 2-truncated simplicial set `S` has an underlying refl quiver with `S _[0]₂` as its underlying
type. -/
def OneTruncation₂ (S : SSet.Truncated 2) := S _[0]₂


/-- Abbreviations for face maps in the 2-truncated simplex category. -/
abbrev δ₂ {n} (i : Fin (n + 2)) (hn := by decide) (hn' := by decide) :
    (⟨[n], hn⟩ : SimplexCategory.Truncated 2) ⟶ ⟨[n + 1], hn'⟩ := SimplexCategory.δ i


/-- Abbreviations for degeneracy maps in the 2-truncated simplex category. -/
abbrev σ₂ {n} (i : Fin (n + 1)) (hn := by decide) (hn' := by decide) :
    (⟨[n+1], hn⟩ : SimplexCategory.Truncated 2) ⟶ ⟨[n], hn'⟩ := SimplexCategory.σ i


@[reassoc (attr := simp)]
                             /-
                               ⊢ LE.le (SimplexCategory.mk 0).len 2
                             -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
                                              /-
                                                🎉 no goals
                                              -/
lemma δ₂_zero_comp_σ₂_zero : δ₂ (0 : Fin 2) ≫ σ₂ 0 = 𝟙 _ := SimplexCategory.δ_comp_σ_self
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc (attr := simp)]
                            /-
                              ⊢ LE.le (SimplexCategory.mk 0).len 2
                            -/
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                                             /-
                                               🎉 no goals
                                             -/
lemma δ₂_one_comp_σ₂_zero : δ₂ (1 : Fin 2) ≫ σ₂ 0 = 𝟙 _ := SimplexCategory.δ_comp_σ_succ
                                             /-
                                               🎉 no goals
                                             -/


/-- The hom-types of the refl quiver underlying a simplicial set `S` are types of edges in `S _[1]₂`
together with source and target equalities. -/
@[ext]
structure OneTruncation₂.Hom {S : SSet.Truncated 2} (X Y : OneTruncation₂ S) where
  /-- An arrow in `OneTruncation₂.Hom X Y` includes the data of a 1-simplex. -/
  edge : S _[1]₂
  /-- An arrow in `OneTruncation₂.Hom X Y` includes a source equality. -/
                  /-
                    S : SSet.Truncated 2
                    X Y : SSet.OneTruncation₂ S
                    edge : S.obj { unop := { obj := SimplexCategory.mk 1, property := ⋯ } }
                    ⊢ LE.le (SimplexCategory.mk 0).len 2
                  -/
                  /-
                    🎉 no goals
                  -/
  src_eq : S.map (δ₂ 1).op edge = X
                  /-
                    🎉 no goals
                  -/
  /-- An arrow in `OneTruncation₂.Hom X Y` includes a target equality. -/
                  /-
                    S : SSet.Truncated 2
                    X Y : SSet.OneTruncation₂ S
                    edge : S.obj { unop := { obj := SimplexCategory.mk 1, property := ⋯ } }
                    src_eq : Eq (S.map (SSet.δ₂ 1 ⋯ ⋯).op edge) X
                    ⊢ LE.le (SimplexCategory.mk 0).len 2
                  -/
                  /-
                    🎉 no goals
                  -/
  tgt_eq : S.map (δ₂ 0).op edge = Y
                  /-
                    🎉 no goals
                  -/


/-- A 2-truncated simplicial set `S` has an underlying refl quiver `SSet.OneTruncation₂ S`. -/
instance (S : SSet.Truncated 2) : ReflQuiver (OneTruncation₂ S) where
  Hom X Y := SSet.OneTruncation₂.Hom X Y
  id X :=
                     /-
                       S : SSet.Truncated 2
                       X : SSet.OneTruncation₂ S
                       ⊢ LE.le (SimplexCategory.mk (HAdd.hAdd 0 1)).len 2
                     -/
                     /-
                       🎉 no goals
                     -/
    { edge := S.map (SSet.σ₂ (n := 0) 0).op X
                     /-
                       🎉 no goals
                     -/
      src_eq := by
        simp only [← FunctorToTypes.map_comp_apply, ← op_comp, δ₂_one_comp_σ₂_zero,
          op_id, FunctorToTypes.map_id_apply]
      tgt_eq := by
        simp only [← FunctorToTypes.map_comp_apply, ← op_comp, δ₂_zero_comp_σ₂_zero,
          op_id, FunctorToTypes.map_id_apply] }


@[simp]
lemma OneTruncation₂.id_edge {S : SSet.Truncated 2} (X : OneTruncation₂ S) :
                                             /-
                                               S : SSet.Truncated 2
                                               X : SSet.OneTruncation₂ S
                                               ⊢ LE.le (SimplexCategory.mk (HAdd.hAdd 0 1)).len 2
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
    OneTruncation₂.Hom.edge (𝟙rq X) = S.map (SSet.σ₂ 0).op X := rfl
                                             /-
                                               🎉 no goals
                                             -/


/-- The functor that carries a 2-truncated simplicial set to its underlying refl quiver. -/
@[simps]
def oneTruncation₂ : SSet.Truncated.{u} 2 ⥤ ReflQuiv.{u, u} where
  obj S := ReflQuiv.of (OneTruncation₂ S)
  map {S T} F := {
    obj := F.app (op [0]₂)
    map := fun f ↦
      { edge := F.app _ f.edge
                     /-
                       S T : SSet.Truncated 2
                       F : Quiver.Hom S T
                       X✝ Y✝ : ↑((fun S => CategoryTheory.ReflQuiv.of (SSet.OneTruncation₂ S)) S)
                       f : Quiver.Hom X✝ Y✝
                       ⊢ Eq (T.map (SSet.δ₂ 1 ⋯ ⋯).op (F.app { unop := { obj := SimplexCategory.mk 1, …
                     -/
        src_eq := by rw [← FunctorToTypes.naturality, f.src_eq]
                     /-
                       🎉 no goals
                     -/
                     /-
                       S T : SSet.Truncated 2
                       F : Quiver.Hom S T
                       X✝ Y✝ : ↑((fun S => CategoryTheory.ReflQuiv.of (SSet.OneTruncation₂ S)) S)
                       f : Quiver.Hom X✝ Y✝
                       ⊢ Eq (T.map (SSet.δ₂ 0 ⋯ ⋯).op (F.app { unop := { obj := SimplexCategory.mk 1, …
                     -/
        tgt_eq := by rw [← FunctorToTypes.naturality, f.tgt_eq] }
                     /-
                       🎉 no goals
                     -/
    map_id := fun X ↦ OneTruncation₂.Hom.ext (by
      /-
        S T : SSet.Truncated 2
        F : Quiver.Hom S T
        X : ↑((fun S => CategoryTheory.ReflQuiv.of (SSet.OneTruncation₂ S)) S)
        ⊢ Eq ({ obj := F.app { unop := { obj := SimplexCategory.mk 0, property := ⋯ }  …
      -/
      dsimp
      /-
        S T : SSet.Truncated 2
        F : Quiver.Hom S T
        X : ↑((fun S => CategoryTheory.ReflQuiv.of (SSet.OneTruncation₂ S)) S)
        ⊢ Eq (F.app { unop := { obj := SimplexCategory.mk 1, property := ⋯ } } (S.map  …
      -/
      rw [← FunctorToTypes.naturality]) }
      /-
        🎉 no goals
      -/


@[ext]
lemma OneTruncation₂.hom_ext {S : SSet.Truncated 2} {x y : OneTruncation₂ S} {f g : x ⟶ y} :
    f.edge = g.edge → f = g := OneTruncation₂.Hom.ext


/-- An equivalence between the type of objects underlying a category and the type of 0-simplices in
the 2-truncated nerve. -/
@[simps]
def OneTruncation₂.nerveEquiv :
    OneTruncation₂ ((SSet.truncation 2).obj (nerve C)) ≃ C where
             /-
               C : Type u
               inst✝ : CategoryTheory.Category.{v, u} C
               X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve C))
               ⊢ LE.le 0 (Opposite.unop ((SimplexCategory.Truncated.inclusion 2).op.obj { uno …
             -/
  toFun X := X.obj' 0
             /-
               🎉 no goals
             -/
  invFun X := .mk₀ X
  left_inv _ := ComposableArrows.ext₀ rfl
  right_inv _ := rfl


/-- A hom equivalence over the function `OneTruncation₂.nerveEquiv`. -/
def OneTruncation₂.nerveHomEquiv (X Y : OneTruncation₂ ((SSet.truncation 2).obj (nerve C))) :
  (X ⟶ Y) ≃ (nerveEquiv X ⟶ nerveEquiv Y) where
  toFun φ := eqToHom (congr_arg ComposableArrows.left φ.src_eq.symm) ≫ φ.edge.hom ≫
      eqToHom (congr_arg ComposableArrows.left φ.tgt_eq)
  invFun f :=
    { edge := ComposableArrows.mk₁ f
      src_eq := ComposableArrows.ext₀ rfl
      tgt_eq := ComposableArrows.ext₀ rfl }
  left_inv φ := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      X Y : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve C))
      φ : Quiver.Hom X Y
      ⊢ Eq ((fun f => { edge := CategoryTheory.ComposableArrows.mk₁ f, src_eq := ⋯,  …
    -/
    ext
    exact ComposableArrows.ext₁ (congr_arg ComposableArrows.left φ.src_eq).symm
      (congr_arg ComposableArrows.left φ.tgt_eq).symm rfl
                    /-
                      C : Type u
                      inst✝ : CategoryTheory.Category.{v, u} C
                      X Y : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve C))
                      f : Quiver.Hom (SSet.OneTruncation₂.nerveEquiv X) (SSet.OneTruncation₂.nerveEq …
                      ⊢ Eq ((fun φ => CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
                    -/
  right_inv f := by dsimp; simp only [comp_id, id_comp]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- The refl quiver underlying a nerve is isomorphic to the refl quiver underlying the category. -/
def OneTruncation₂.ofNerve₂ (C : Type u) [Category.{u} C] :
    ReflQuiv.of (OneTruncation₂ (nerveFunctor₂.obj (Cat.of C))) ≅ ReflQuiv.of C := by
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    ⊢ CategoryTheory.Iso (CategoryTheory.ReflQuiv.of (SSet.OneTruncation₂ (Categor …
  -/
  apply ReflQuiv.isoOfEquiv.{u,u} OneTruncation₂.nerveEquiv OneTruncation₂.nerveHomEquiv ?_
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    ⊢ ∀ (X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve ↑( …
  -/
  intro X
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve ↑(Categ …
    ⊢ Eq ((X.nerveHomEquiv X) (CategoryTheory.ReflQuiver.id X)) (CategoryTheory.Re …
  -/
  unfold nerveEquiv nerveHomEquiv
  simp only [Cat.of_α, op_obj, ComposableArrows.obj', Fin.zero_eta, Fin.isValue, Equiv.coe_fn_mk,
    nerveEquiv_apply, Nat.reduceAdd, id_edge, SimplexCategory.len_mk, id_eq, eqToHom_refl, comp_id,
    id_comp, ReflQuiver.id_eq_id]
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve ↑(Categ …
    ⊢ Eq (CategoryTheory.ComposableArrows.hom (((SSet.truncation 2).obj (CategoryT …
  -/
  unfold nerve truncation SimplicialObject.truncation SimplexCategory.Truncated.inclusion
  simp only [fullSubcategoryInclusion.obj, SimplexCategory.len_mk, Nat.reduceAdd, Fin.isValue,
    SimplexCategory.toCat_map, whiskeringLeft_obj_obj, Functor.comp_map, op_obj, op_map,
    Quiver.Hom.unop_op, fullSubcategoryInclusion.map, ComposableArrows.whiskerLeft_map,
    Fin.zero_eta, Monotone.functor_obj, Fin.mk_one, homOfLE_leOfHom]
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve ↑(Categ …
    ⊢ Eq (X.map (⋯.functor.map (CategoryTheory.homOfLE ⋯))) (CategoryTheory.Catego …
  -/
  show X.map (𝟙 _) = _
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve ↑(Categ …
    ⊢ Eq (X.map (CategoryTheory.CategoryStruct.id (((CategoryTheory.forget₂ PartOr …
  -/
  rw [X.map_id]
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    C : Type u
    inst✝ : CategoryTheory.Category.{u, u} C
    X : SSet.OneTruncation₂ ((SSet.truncation 2).obj (CategoryTheory.nerve ↑(Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (X.obj (((CategoryTheory.forget₂ PartOr …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The refl quiver underlying a nerve is naturally isomorphic to the refl quiver underlying the
category. -/
@[simps! hom_app_obj hom_app_map inv_app_obj_obj inv_app_obj_map inv_app_map]
def OneTruncation₂.ofNerve₂.natIso :
    nerveFunctor₂.{u,u} ⋙ SSet.oneTruncation₂ ≅ ReflQuiv.forget :=
  NatIso.ofComponents (fun C => OneTruncation₂.ofNerve₂ C) (by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      ⊢ ∀ {X Y : CategoryTheory.Cat} (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
    -/
  · intro C D F
    /-
      C✝ : Type u
      inst✝ : CategoryTheory.Category.{v, u} C✝
      C D : CategoryTheory.Cat
      F : Quiver.Hom C D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Nerve.nerveFunctor₂. …
    -/
    fapply ReflPrefunctor.ext <;> simp
      /-
        case h_obj
        C✝ : Type u
        inst✝ : CategoryTheory.Category.{v, u} C✝
        C D : CategoryTheory.Cat
        F : Quiver.Hom C D
        ⊢ ∀ (X : SSet.OneTruncation₂ (CategoryTheory.Nerve.nerveFunctor₂.obj C)), Eq ( …
      -/
    · exact fun _ ↦ rfl
      /-
        🎉 no goals
      -/
      /-
        case h_map
        C✝ : Type u
        inst✝ : CategoryTheory.Category.{v, u} C✝
        C D : CategoryTheory.Cat
        F : Quiver.Hom C D
        ⊢ ∀ (X Y : SSet.OneTruncation₂ (CategoryTheory.Nerve.nerveFunctor₂.obj C)) (f  …
      -/
    · intro X Y f
      /-
        case h_map
        C✝ : Type u
        inst✝ : CategoryTheory.Category.{v, u} C✝
        C D : CategoryTheory.Cat
        F : Quiver.Hom C D
        X Y : SSet.OneTruncation₂ (CategoryTheory.Nerve.nerveFunctor₂.obj C)
        f : Quiver.Hom X Y
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (SSet.oneTruncation₂.map (CategoryTh …
      -/
      obtain ⟨f, rfl, rfl⟩ := f
      unfold SSet.oneTruncation₂ nerveFunctor₂ SSet.truncation SimplicialObject.truncation
        nerveFunctor toReflPrefunctor
      simp only [comp_obj, whiskeringLeft_obj_obj, ReflQuiv.of_val, Functor.comp_map,
        whiskeringLeft_obj_map, whiskerLeft_app, op_obj, whiskeringRight_obj_obj, ofNerve₂,
        Cat.of_α, nerveEquiv, ComposableArrows.obj', Fin.zero_eta, Fin.isValue,
        ReflQuiv.comp_eq_comp, Nat.reduceAdd, SimplexCategory.len_mk, id_eq, op_map,
        Quiver.Hom.unop_op, nerve_map, SimplexCategory.toCat_map, ReflPrefunctor.comp_obj,
        ReflPrefunctor.comp_map]
      /-
        case h_map.mk
        C✝ : Type u
        inst✝ : CategoryTheory.Category.{v, u} C✝
        C D : CategoryTheory.Cat
        F : Quiver.Hom C D
        f : (CategoryTheory.Nerve.nerveFunctor₂.obj C).obj { unop := { obj := SimplexC …
        ⊢ Eq ((CategoryTheory.ReflQuiv.isoOfEquiv { toFun := fun X => X.obj 0, invFun  …
      -/
      simp [nerveHomEquiv, ReflQuiv.isoOfEquiv, ReflQuiv.isoOfQuivIso, Quiv.isoOfEquiv])
      /-
        🎉 no goals
      -/


private lemma map_map_of_eq.{w}  {C : Type u} [Category.{v} C] (V : Cᵒᵖ ⥤ Type w) {X Y Z : C}
    {α : X ⟶ Y} {β : Y ⟶ Z} {γ : X ⟶ Z} {φ} :
    α ≫ β = γ → V.map α.op (V.map β.op φ) = V.map γ.op φ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    V : CategoryTheory.Functor (Opposite C) (Type w)
    X Y Z : C
    α : Quiver.Hom X Y
    β : Quiver.Hom Y Z
    γ : Quiver.Hom X Z
    φ : V.obj { unop := Z }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α β) γ → Eq (V.map α.op (V.map β.op φ …
  -/
  rintro rfl
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    V : CategoryTheory.Functor (Opposite C) (Type w)
    X Y Z : C
    α : Quiver.Hom X Y
    β : Quiver.Hom Y Z
    φ : V.obj { unop := Z }
    ⊢ Eq (V.map α.op (V.map β.op φ)) (V.map (CategoryTheory.CategoryStruct.comp α  …
  -/
  change (V.map _ ≫ V.map _) _ = _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    V : CategoryTheory.Functor (Opposite C) (Type w)
    X Y Z : C
    α : Quiver.Hom X Y
    β : Quiver.Hom Y Z
    φ : V.obj { unop := Z }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (V.map β.op) (V.map α.op) φ) (V.map ( …
  -/
  rw [← map_comp]; rfl
                   /-
                     🎉 no goals
                   -/


local notation (priority := high) "[" n "]" => SimplexCategory.mk n


/-- The map that picks up the initial vertex of a 2-simplex, as a morphism in the 2-truncated
simplex category. -/
                         /-
                           V : SSet
                           ⊢ LE.le (SimplexCategory.mk 0).len 2
                         -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
                                         /-
                                           🎉 no goals
                                         -/
def ι0₂ : [0]₂ ⟶ [2]₂ := δ₂ (n := 0) 1 ≫ δ₂ (n := 1) 1
                                         /-
                                           🎉 no goals
                                         -/


/-- The map that picks up the middle vertex of a 2-simplex, as a morphism in the 2-truncated
simplex category. -/
                         /-
                           V : SSet
                           ⊢ LE.le (SimplexCategory.mk 0).len 2
                         -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
                                         /-
                                           🎉 no goals
                                         -/
def ι1₂ : [0]₂ ⟶ [2]₂ := δ₂ (n := 0) 0 ≫ δ₂ (n := 1) 2
                                         /-
                                           🎉 no goals
                                         -/


/-- The map that picks up the final vertex of a 2-simplex, as a morphism in the 2-truncated
simplex category. -/
                         /-
                           V : SSet
                           ⊢ LE.le (SimplexCategory.mk 0).len 2
                         -/
                         /-
                           🎉 no goals
                         -/
                         /-
                           🎉 no goals
                         -/
                                         /-
                                           🎉 no goals
                                         -/
def ι2₂ : [0]₂ ⟶ [2]₂ := δ₂ (n := 0) 0 ≫ δ₂ (n := 1) 1
                                         /-
                                           🎉 no goals
                                         -/


/-- The initial vertex of a 2-simplex in a 2-truncated simplicial set. -/
def ev0₂ {V : SSet.Truncated 2} (φ : V _[2]₂) : OneTruncation₂ V := V.map ι0₂.op φ


/-- The middle vertex of a 2-simplex in a 2-truncated simplicial set. -/
def ev1₂ {V : SSet.Truncated 2} (φ : V _[2]₂) : OneTruncation₂ V := V.map ι1₂.op φ


/-- The final vertex of a 2-simplex in a 2-truncated simplicial set. -/
def ev2₂ {V : SSet.Truncated 2} (φ : V _[2]₂) : OneTruncation₂ V := V.map ι2₂.op φ


/-- The 0th face of a 2-simplex, as a morphism in the 2-truncated simplex category. -/
                         /-
                           V : SSet
                           ⊢ LE.le (SimplexCategory.mk 1).len 2
                         -/
                         /-
                           🎉 no goals
                         -/
def δ0₂ : [1]₂ ⟶ [2]₂ := δ₂ (n := 1) 0
                         /-
                           🎉 no goals
                         -/


/-- The 1st face of a 2-simplex, as a morphism in the 2-truncated simplex category. -/
                         /-
                           V : SSet
                           ⊢ LE.le (SimplexCategory.mk 1).len 2
                         -/
                         /-
                           🎉 no goals
                         -/
def δ1₂ : [1]₂ ⟶ [2]₂ := δ₂ (n := 1) 1
                         /-
                           🎉 no goals
                         -/


/-- The 2nd face of a 2-simplex, as a morphism in the 2-truncated simplex category. -/
                         /-
                           V : SSet
                           ⊢ LE.le (SimplexCategory.mk 1).len 2
                         -/
                         /-
                           🎉 no goals
                         -/
def δ2₂ : [1]₂ ⟶ [2]₂ := δ₂ (n := 1) 2
                         /-
                           🎉 no goals
                         -/


/-- The arrow in the ReflQuiver `OneTruncation₂ V` of a 2-truncated simplicial set arising from the
0th face of a 2-simplex. -/
def ev12₂ {V : SSet.Truncated 2} (φ : V _[2]₂) : ev1₂ φ ⟶ ev2₂ φ :=
  ⟨V.map δ0₂.op φ,
                                                                    /-
                                                                      V✝ : SSet
                                                                      V : SSet.Truncated 2
                                                                      φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
                                                                      ⊢ LE.le 0 1
                                                                    -/
    map_map_of_eq V (SimplexCategory.δ_comp_δ (i := 0) (j := 1) (by decide)).symm,
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    map_map_of_eq V rfl⟩


/-- The arrow in the ReflQuiver `OneTruncation₂ V` of a 2-truncated simplicial set arising from the
1st face of a 2-simplex. -/
def ev02₂ {V : SSet.Truncated 2} (φ : V _[2]₂) : ev0₂ φ ⟶ ev2₂ φ :=
  ⟨V.map δ1₂.op φ, map_map_of_eq V rfl, map_map_of_eq V rfl⟩


/-- The arrow in the ReflQuiver `OneTruncation₂ V` of a 2-truncated simplicial set arising from the
2nd face of a 2-simplex. -/
def ev01₂ {V : SSet.Truncated 2} (φ : V _[2]₂) : ev0₂ φ ⟶ ev1₂ φ :=
  ⟨V.map δ2₂.op φ, map_map_of_eq V (SimplexCategory.δ_comp_δ (j := 1) le_rfl), map_map_of_eq V rfl⟩



/-- The 2-simplices in a 2-truncated simplicial set `V` generate a hom relation on the free
category on the underlying refl quiver of `V`. -/
inductive HoRel₂ {V : SSet.Truncated 2} :
    (X Y : Cat.FreeRefl (OneTruncation₂ V)) → (f g : X ⟶ Y) → Prop
  | mk (φ : V _[2]₂) :
    HoRel₂ _ _
      (Quot.mk _ (Quiver.Hom.toPath (ev02₂ φ)))
      (Quot.mk _ ((Quiver.Hom.toPath (ev01₂ φ)).comp
        (Quiver.Hom.toPath (ev12₂ φ))))


/-- A 2-simplex whose faces are identified with certain arrows in `OneTruncation₂ V` defines
a term of type `HoRel₂` between those arrows. -/
theorem HoRel₂.mk' {V : SSet.Truncated 2} (φ : V _[2]₂) {X₀ X₁ X₂ : OneTruncation₂ V}
    (f₀₁ : X₀ ⟶ X₁) (f₁₂ : X₁ ⟶ X₂) (f₀₂ : X₀ ⟶ X₂)
                             /-
                               V✝ : SSet
                               V : SSet.Truncated 2
                               φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
                               X₀ X₁ X₂ : SSet.OneTruncation₂ V
                               f₀₁ : Quiver.Hom X₀ X₁
                               f₁₂ : Quiver.Hom X₁ X₂
                               f₀₂ : Quiver.Hom X₀ X₂
                               ⊢ LE.le (SimplexCategory.mk 1).len 2
                             -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
    (h₀₁ : f₀₁.edge = V.map (δ₂ 2).op φ) (h₁₂ : f₁₂.edge = V.map (δ₂ 0).op φ)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                             /-
                               V✝ : SSet
                               V : SSet.Truncated 2
                               φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
                               X₀ X₁ X₂ : SSet.OneTruncation₂ V
                               f₀₁ : Quiver.Hom X₀ X₁
                               f₁₂ : Quiver.Hom X₁ X₂
                               f₀₂ : Quiver.Hom X₀ X₂
                               h₀₁ : Eq f₀₁.edge (V.map (SSet.δ₂ 2 ⋯ ⋯).op φ)
                               h₁₂ : Eq f₁₂.edge (V.map (SSet.δ₂ 0 ⋯ ⋯).op φ)
                               ⊢ LE.le (SimplexCategory.mk 1).len 2
                             -/
                             /-
                               🎉 no goals
                             -/
    (h₀₂ : f₀₂.edge = V.map (δ₂ 1).op φ) :
                             /-
                               🎉 no goals
                             -/
    HoRel₂ _ _ (Quot.mk _ (Quiver.Hom.toPath f₀₂))
      (Quot.mk _ ((Quiver.Hom.toPath f₀₁).comp (Quiver.Hom.toPath f₁₂))) := by
  obtain rfl : X₀ = ev0₂ φ := by
    rw [← f₀₂.src_eq, h₀₂, ← FunctorToTypes.map_comp_apply, ← op_comp]
    rfl
  obtain rfl : X₁ = ev1₂ φ := by
    rw [← f₀₁.tgt_eq, h₀₁, ← FunctorToTypes.map_comp_apply, ← op_comp]
    rfl
  obtain rfl : X₂ = ev2₂ φ := by
    rw [← f₁₂.tgt_eq, h₁₂, ← FunctorToTypes.map_comp_apply, ← op_comp]
    rfl
  /-
    V : SSet.Truncated 2
    φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
    f₀₁ : Quiver.Hom (SSet.Truncated.ev0₂ φ) (SSet.Truncated.ev1₂ φ)
    h₀₁ : Eq f₀₁.edge (V.map (SSet.δ₂ 2 ⋯ ⋯).op φ)
    f₀₂ : Quiver.Hom (SSet.Truncated.ev0₂ φ) (SSet.Truncated.ev2₂ φ)
    h₀₂ : Eq f₀₂.edge (V.map (SSet.δ₂ 1 ⋯ ⋯).op φ)
    f₁₂ : Quiver.Hom (SSet.Truncated.ev1₂ φ) (SSet.Truncated.ev2₂ φ)
    h₁₂ : Eq f₁₂.edge (V.map (SSet.δ₂ 0 ⋯ ⋯).op φ)
    ⊢ SSet.Truncated.HoRel₂ { as := SSet.Truncated.ev0₂ φ } { as := SSet.Truncated …
  -/
  obtain rfl : f₀₁ = ev01₂ φ := by ext; assumption
  /-
    V : SSet.Truncated 2
    φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
    f₀₂ : Quiver.Hom (SSet.Truncated.ev0₂ φ) (SSet.Truncated.ev2₂ φ)
    h₀₂ : Eq f₀₂.edge (V.map (SSet.δ₂ 1 ⋯ ⋯).op φ)
    f₁₂ : Quiver.Hom (SSet.Truncated.ev1₂ φ) (SSet.Truncated.ev2₂ φ)
    h₁₂ : Eq f₁₂.edge (V.map (SSet.δ₂ 0 ⋯ ⋯).op φ)
    h₀₁ : Eq (SSet.Truncated.ev01₂ φ).edge (V.map (SSet.δ₂ 2 ⋯ ⋯).op φ)
    ⊢ SSet.Truncated.HoRel₂ { as := SSet.Truncated.ev0₂ φ } { as := SSet.Truncated …
  -/
  obtain rfl : f₁₂ = ev12₂ φ := by ext; assumption
  /-
    V : SSet.Truncated 2
    φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
    f₀₂ : Quiver.Hom (SSet.Truncated.ev0₂ φ) (SSet.Truncated.ev2₂ φ)
    h₀₂ : Eq f₀₂.edge (V.map (SSet.δ₂ 1 ⋯ ⋯).op φ)
    h₀₁ : Eq (SSet.Truncated.ev01₂ φ).edge (V.map (SSet.δ₂ 2 ⋯ ⋯).op φ)
    h₁₂ : Eq (SSet.Truncated.ev12₂ φ).edge (V.map (SSet.δ₂ 0 ⋯ ⋯).op φ)
    ⊢ SSet.Truncated.HoRel₂ { as := SSet.Truncated.ev0₂ φ } { as := SSet.Truncated …
  -/
  obtain rfl : f₀₂ = ev02₂ φ := by ext; assumption
  /-
    V : SSet.Truncated 2
    φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
    h₀₁ : Eq (SSet.Truncated.ev01₂ φ).edge (V.map (SSet.δ₂ 2 ⋯ ⋯).op φ)
    h₁₂ : Eq (SSet.Truncated.ev12₂ φ).edge (V.map (SSet.δ₂ 0 ⋯ ⋯).op φ)
    h₀₂ : Eq (SSet.Truncated.ev02₂ φ).edge (V.map (SSet.δ₂ 1 ⋯ ⋯).op φ)
    ⊢ SSet.Truncated.HoRel₂ { as := SSet.Truncated.ev0₂ φ } { as := SSet.Truncated …
  -/
  constructor
  /-
    🎉 no goals
  -/


/-- The type underlying the homotopy category of a 2-truncated simplicial set `V`. -/
def _root_.SSet.Truncated.HomotopyCategory (V : SSet.Truncated.{u} 2) : Type u :=
  Quotient (HoRel₂ (V := V))


instance (V : SSet.Truncated.{u} 2) : Category.{u} (V.HomotopyCategory) :=
  inferInstanceAs (Category (CategoryTheory.Quotient ..))


/-- A canonical functor from the free category on the refl quiver underlying a 2-truncated
simplicial set `V` to its homotopy category. -/
def _root_.SSet.Truncated.HomotopyCategory.quotientFunctor (V : SSet.Truncated.{u} 2) :
    Cat.FreeRefl (OneTruncation₂ V) ⥤ V.HomotopyCategory :=
  Quotient.functor _


/-- By `Quotient.lift_unique'` (not `Quotient.lift`) we have that `quotientFunctor V` is an
epimorphism. -/
theorem HomotopyCategory.lift_unique' (V : SSet.Truncated.{u} 2) {D} [Category D]
    (F₁ F₂ : V.HomotopyCategory ⥤ D)
    (h : HomotopyCategory.quotientFunctor V ⋙ F₁ = HomotopyCategory.quotientFunctor V ⋙ F₂) :
    F₁ = F₂ :=
  Quotient.lift_unique' (C := Cat.FreeRefl (OneTruncation₂ V))
    (HoRel₂ (V := V)) _ _ h


/-- A map of 2-truncated simplicial sets induces a functor between homotopy categories. -/
def mapHomotopyCategory {V W : SSet.Truncated.{u} 2} (F : V ⟶ W) :
    V.HomotopyCategory ⥤ W.HomotopyCategory :=
  CategoryTheory.Quotient.lift _
    ((oneTruncation₂ ⋙ Cat.freeRefl).map F ⋙ HomotopyCategory.quotientFunctor W)
    (by
      /-
        V✝ : SSet
        V W : SSet.Truncated 2
        F : Quiver.Hom V W
        ⊢ ∀ (x y : CategoryTheory.Cat.FreeRefl (SSet.OneTruncation₂ V)) (f₁ f₂ : Quive …
      -/
      rintro _ _ _ _ ⟨φ⟩
      /-
        case mk
        V✝ : SSet
        V W : SSet.Truncated 2
        F : Quiver.Hom V W
        X Y : CategoryTheory.Cat.FreeRefl (SSet.OneTruncation₂ V)
        φ : V.obj { unop := { obj := SimplexCategory.mk 2, property := ⋯ } }
        ⊢ Eq ((CategoryTheory.Functor.comp ((SSet.oneTruncation₂.comp CategoryTheory.C …
      -/
      apply CategoryTheory.Quotient.sound
      apply HoRel₂.mk' (φ := F.app _ φ)
        (f₀₁ := (oneTruncation₂.map F).map (ev01₂ φ))
        (f₀₂ := (oneTruncation₂.map F).map (ev02₂ φ))
        (f₁₂ := (oneTruncation₂.map F).map (ev12₂ φ))
      all_goals
        apply FunctorToTypes.naturality)


/-- The functor that takes a 2-truncated simplicial set to its homotopy category. -/
def hoFunctor₂ : SSet.Truncated.{u} 2 ⥤ Cat.{u,u} where
  obj V := Cat.of (V.HomotopyCategory)
  map {S T} F := mapHomotopyCategory F
  map_id S := by
    /-
      V : SSet
      S : SSet.Truncated 2
      ⊢ Eq ({ obj := fun V => CategoryTheory.Cat.of V.HomotopyCategory, map := fun { …
    -/
    apply Quotient.lift_unique'
    /-
      case h
      V : SSet
      S : SSet.Truncated 2
      ⊢ Eq ((CategoryTheory.Quotient.functor SSet.Truncated.HoRel₂).comp ({ obj := f …
    -/
    simp [mapHomotopyCategory, Quotient.lift_spec]
    /-
      case h
      V : SSet
      S : SSet.Truncated 2
      ⊢ Eq (CategoryTheory.Functor.comp (CategoryTheory.CategoryStruct.id (CategoryT …
    -/
    exact Eq.trans (Functor.id_comp ..) (Functor.comp_id _).symm
    /-
      🎉 no goals
    -/
  map_comp {S T U} F G := by
    /-
      V : SSet
      S T U : SSet.Truncated 2
      F : Quiver.Hom S T
      G : Quiver.Hom T U
      ⊢ Eq ({ obj := fun V => CategoryTheory.Cat.of V.HomotopyCategory, map := fun { …
    -/
    apply Quotient.lift_unique'
    /-
      case h
      V : SSet
      S T U : SSet.Truncated 2
      F : Quiver.Hom S T
      G : Quiver.Hom T U
      ⊢ Eq ((CategoryTheory.Quotient.functor SSet.Truncated.HoRel₂).comp ({ obj := f …
    -/
    simp [mapHomotopyCategory, SSet.Truncated.HomotopyCategory.quotientFunctor]
    rw [Quotient.lift_spec, Cat.comp_eq_comp, Cat.comp_eq_comp, ← Functor.assoc, Functor.assoc,
      Quotient.lift_spec, Functor.assoc, Quotient.lift_spec]


theorem hoFunctor₂_naturality {X Y : SSet.Truncated.{u} 2} (f : X ⟶ Y) :
    (oneTruncation₂ ⋙ Cat.freeRefl).map f ⋙ SSet.Truncated.HomotopyCategory.quotientFunctor Y =
      SSet.Truncated.HomotopyCategory.quotientFunctor X ⋙ mapHomotopyCategory f := rfl


/-- The functor that takes a simplicial set to its homotopy category by passing through the
2-truncation. -/
def hoFunctor : SSet.{u} ⥤ Cat.{u, u} := SSet.truncation 2 ⋙ Truncated.hoFunctor₂


