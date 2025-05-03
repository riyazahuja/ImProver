/-- A possibly empty category for which every functor to a discrete category is constant.
-/
class IsPreconnected (J : Type u₁) [Category.{v₁} J] : Prop where
  iso_constant :
    ∀ {α : Type u₁} (F : J ⥤ Discrete α) (j : J), Nonempty (F ≅ (Functor.const J).obj (F.obj j))


/-- We define a connected category as a _nonempty_ category for which every
functor to a discrete category is constant.

NB. Some authors include the empty category as connected, we do not.
We instead are interested in categories with exactly one 'connected
component'.

This allows us to show that the functor X ⨯ - preserves connected limits.

See <https://stacks.math.columbia.edu/tag/002S>
-/
class IsConnected (J : Type u₁) [Category.{v₁} J] extends IsPreconnected J : Prop where
  [is_nonempty : Nonempty J]


/-- Implementation detail of `isoConstant`. -/
private def liftToDiscrete {α : Type u₂} (F : J ⥤ Discrete α) : J ⥤ Discrete J where
  obj j := have := Nonempty.intro j
    Discrete.mk (Function.invFun F.obj (F.obj j))
  map {j _} f := have := Nonempty.intro j
    ⟨⟨congr_arg (Function.invFun F.obj) (Discrete.ext (Discrete.eq_of_hom (F.map f)))⟩⟩


/-- Implementation detail of `isoConstant`. -/
private def factorThroughDiscrete {α : Type u₂} (F : J ⥤ Discrete α) :
    liftToDiscrete F ⋙ Discrete.functor F.obj ≅ F :=
                                                                         /-
                                                                           J : Type u₁
                                                                           inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                           K : Type u₂
                                                                           inst✝ : CategoryTheory.Category.{v₂, u₂} K
                                                                           α : Type u₂
                                                                           F : CategoryTheory.Functor J (CategoryTheory.Discrete α)
                                                                           ⊢ ∀ {X Y : J} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                                         -/
  NatIso.ofComponents (fun _ => eqToIso Function.apply_invFun_apply) (by aesop_cat)
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- If `J` is connected, any functor `F : J ⥤ Discrete α` is isomorphic to
the constant functor with value `F.obj j` (for any choice of `j`).
-/
def isoConstant [IsPreconnected J] {α : Type u₂} (F : J ⥤ Discrete α) (j : J) :
    F ≅ (Functor.const J).obj (F.obj j) :=
  (IsPreconnected.IsoConstantAux.factorThroughDiscrete F).symm
    ≪≫ isoWhiskerRight (IsPreconnected.iso_constant _ j).some _
                                                                              /-
                                                                                J : Type u₁
                                                                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                                K : Type u₂
                                                                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                                                inst✝ : CategoryTheory.IsPreconnected J
                                                                                α : Type u₂
                                                                                F : CategoryTheory.Functor J (CategoryTheory.Discrete α)
                                                                                j : J
                                                                                ⊢ ∀ {X Y : J} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
                                                                              -/
    ≪≫ NatIso.ofComponents (fun _ => eqToIso Function.apply_invFun_apply) (by aesop_cat)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- If `J` is connected, any functor to a discrete category is constant on objects.
The converse is given in `IsConnected.of_any_functor_const_on_obj`.
-/
theorem any_functor_const_on_obj [IsPreconnected J] {α : Type u₂} (F : J ⥤ Discrete α) (j j' : J) :
    F.obj j = F.obj j' := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    inst✝ : CategoryTheory.IsPreconnected J
    α : Type u₂
    F : CategoryTheory.Functor J (CategoryTheory.Discrete α)
    j j' : J
    ⊢ Eq (F.obj j) (F.obj j')
  -/
  ext; exact ((isoConstant F j').hom.app j).down.1
       /-
         🎉 no goals
       -/


/-- If any functor to a discrete category is constant on objects, J is connected.
The converse of `any_functor_const_on_obj`.
-/
theorem IsPreconnected.of_any_functor_const_on_obj
    (h : ∀ {α : Type u₁} (F : J ⥤ Discrete α), ∀ j j' : J, F.obj j = F.obj j') :
    IsPreconnected J where
                               /-
                                 J : Type u₁
                                 inst✝ : CategoryTheory.Category.{v₁, u₁} J
                                 h : ∀ {α : Type u₁} (F : CategoryTheory.Functor J (CategoryTheory.Discrete α)) …
                                 α✝ : Type u₁
                                 F : CategoryTheory.Functor J (CategoryTheory.Discrete α✝)
                                 j' : J
                                 ⊢ ∀ {X Y : J} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
                               -/
  iso_constant := fun F j' => ⟨NatIso.ofComponents fun j => eqToIso (h F j j')⟩
                               /-
                                 🎉 no goals
                               -/


/-- If any functor to a discrete category is constant on objects, J is connected.
The converse of `any_functor_const_on_obj`.
-/
theorem IsConnected.of_any_functor_const_on_obj [Nonempty J]
    (h : ∀ {α : Type u₁} (F : J ⥤ Discrete α), ∀ j j' : J, F.obj j = F.obj j') : IsConnected J :=
  { IsPreconnected.of_any_functor_const_on_obj h with }


/-- If `J` is connected, then given any function `F` such that the presence of a
morphism `j₁ ⟶ j₂` implies `F j₁ = F j₂`, we have that `F` is constant.
This can be thought of as a local-to-global property.

The converse is shown in `IsConnected.of_constant_of_preserves_morphisms`
-/
theorem constant_of_preserves_morphisms [IsPreconnected J] {α : Type u₂} (F : J → α)
    (h : ∀ (j₁ j₂ : J) (_ : j₁ ⟶ j₂), F j₁ = F j₂) (j j' : J) : F j = F j' := by
  simpa using
    any_functor_const_on_obj
      { obj := Discrete.mk ∘ F
        map := fun f => eqToHom (by ext; exact h _ _ f) }
      j j'


/-- `J` is connected if: given any function `F : J → α` which is constant for any
`j₁, j₂` for which there is a morphism `j₁ ⟶ j₂`, then `F` is constant.
This can be thought of as a local-to-global property.

The converse of `constant_of_preserves_morphisms`.
-/
theorem IsPreconnected.of_constant_of_preserves_morphisms
    (h : ∀ {α : Type u₁} (F : J → α),
      (∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), F j₁ = F j₂) → ∀ j j' : J, F j = F j') :
    IsPreconnected J :=
  IsPreconnected.of_any_functor_const_on_obj fun F =>
                        /-
                          J : Type u₁
                          inst✝ : CategoryTheory.Category.{v₁, u₁} J
                          h : ∀ {α : Type u₁} (F : J → α), (∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁)  …
                          α✝ : Type u₁
                          F : CategoryTheory.Functor J (CategoryTheory.Discrete α✝)
                          j₁✝ j₂✝ : J
                          f : Quiver.Hom j₁✝ j₂✝
                          ⊢ Eq (F.obj j₁✝) (F.obj j₂✝)
                        -/
    h F.obj fun f => by ext; exact Discrete.eq_of_hom (F.map f)
                             /-
                               🎉 no goals
                             -/


/-- `J` is connected if: given any function `F : J → α` which is constant for any
`j₁, j₂` for which there is a morphism `j₁ ⟶ j₂`, then `F` is constant.
This can be thought of as a local-to-global property.

The converse of `constant_of_preserves_morphisms`.
-/
theorem IsConnected.of_constant_of_preserves_morphisms [Nonempty J]
    (h : ∀ {α : Type u₁} (F : J → α),
      (∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), F j₁ = F j₂) → ∀ j j' : J, F j = F j') :
    IsConnected J :=
  { IsPreconnected.of_constant_of_preserves_morphisms h with }


/-- An inductive-like property for the objects of a connected category.
If the set `p` is nonempty, and `p` is closed under morphisms of `J`,
then `p` contains all of `J`.

The converse is given in `IsConnected.of_induct`.
-/
theorem induct_on_objects [IsPreconnected J] (p : Set J) {j₀ : J} (h0 : j₀ ∈ p)
    (h1 : ∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), j₁ ∈ p ↔ j₂ ∈ p) (j : J) : j ∈ p := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    inst✝ : CategoryTheory.IsPreconnected J
    p : Set J
    j₀ : J
    h0 : Membership.mem p j₀
    h1 : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Iff (Membership.mem p j₁) (Membership.m …
    j : J
    ⊢ Membership.mem p j
  -/
  let aux (j₁ j₂ : J) (f : j₁ ⟶ j₂) := congrArg ULift.up <| (h1 f).eq
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    inst✝ : CategoryTheory.IsPreconnected J
    p : Set J
    j₀ : J
    h0 : Membership.mem p j₀
    h1 : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Iff (Membership.mem p j₁) (Membership.m …
    j : J
    aux : ∀ (j₁ j₂ : J), Quiver.Hom j₁ j₂ → Eq { down := Membership.mem p j₁ } { d …
    ⊢ Membership.mem p j
  -/
  injection constant_of_preserves_morphisms (fun k => ULift.up.{u₁} (k ∈ p)) aux j j₀ with i
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    inst✝ : CategoryTheory.IsPreconnected J
    p : Set J
    j₀ : J
    h0 : Membership.mem p j₀
    h1 : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Iff (Membership.mem p j₁) (Membership.m …
    j : J
    aux : ∀ (j₁ j₂ : J), Quiver.Hom j₁ j₂ → Eq { down := Membership.mem p j₁ } { d …
    i : Eq (Membership.mem p j) (Membership.mem p j₀)
    ⊢ Membership.mem p j
  -/
  rwa [i]
  /-
    🎉 no goals
  -/


/--
If any maximal connected component containing some element j₀ of J is all of J, then J is connected.

The converse of `induct_on_objects`.
-/
theorem IsConnected.of_induct {j₀ : J}
    (h : ∀ p : Set J, j₀ ∈ p → (∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), j₁ ∈ p ↔ j₂ ∈ p) → ∀ j : J, j ∈ p) :
    IsConnected J :=
  have := Nonempty.intro j₀
  IsConnected.of_constant_of_preserves_morphisms fun {α} F a => by
    have w := h { j | F j = F j₀ } rfl (fun {j₁} {j₂} f => by
      change F j₁ = F j₀ ↔ F j₂ = F j₀
      simp [a f])
    /-
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      j₀ : J
      h : ∀ (p : Set J), Membership.mem p j₀ → (∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → If …
      this : Nonempty J
      α : Type u₁
      F : J → α
      a : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
      w : ∀ (j : J), Membership.mem (setOf fun j => Eq (F j) (F j₀)) j
      ⊢ ∀ (j j' : J), Eq (F j) (F j')
    -/
    intro j j'
    /-
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      j₀ : J
      h : ∀ (p : Set J), Membership.mem p j₀ → (∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → If …
      this : Nonempty J
      α : Type u₁
      F : J → α
      a : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
      w : ∀ (j : J), Membership.mem (setOf fun j => Eq (F j) (F j₀)) j
      j j' : J
      ⊢ Eq (F j) (F j')
    -/
    rw [w j, w j']
    /-
      🎉 no goals
    -/


/-- Lifting the universe level of morphisms and objects preserves connectedness. -/
instance [hc : IsConnected J] : IsConnected (ULiftHom.{v₂} (ULift.{u₂} J)) := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} K
    hc : CategoryTheory.IsConnected J
    ⊢ CategoryTheory.IsConnected (CategoryTheory.ULiftHom (ULift.{u₂, u₁} J))
  -/
  apply IsConnected.of_induct
    /-
      case h
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      hc : CategoryTheory.IsConnected J
      ⊢ ∀ (p : Set (CategoryTheory.ULiftHom (ULift.{u₂, u₁} J))), Membership.mem p ? …
    -/
  · rintro p hj₀ h ⟨j⟩
    /-
      case h.up
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      hc : CategoryTheory.IsConnected J
      p : Set (CategoryTheory.ULiftHom (ULift.{u₂, u₁} J))
      hj₀ : Membership.mem p ?j₀
      h : ∀ {j₁ j₂ : CategoryTheory.ULiftHom (ULift.{u₂, u₁} J)}, Quiver.Hom j₁ j₂ → …
      j : J
      ⊢ Membership.mem p { down := j }
    -/
    let p' : Set J := {j : J | p ⟨j⟩}
    have hj₀' : Classical.choice hc.is_nonempty ∈ p' := by
      simp only [p', (eq_self p')]
      exact hj₀
    /-
      case h.up
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝ : CategoryTheory.Category.{v₂, u₂} K
      hc : CategoryTheory.IsConnected J
      p : Set (CategoryTheory.ULiftHom (ULift.{u₂, u₁} J))
      hj₀ : Membership.mem p { down := Classical.choice ⋯ }
      h : ∀ {j₁ j₂ : CategoryTheory.ULiftHom (ULift.{u₂, u₁} J)}, Quiver.Hom j₁ j₂ → …
      j : J
      p' : Set J := setOf fun j => p { down := j }
      hj₀' : Membership.mem p' (Classical.choice ⋯)
      ⊢ Membership.mem p { down := j }
    -/
    apply induct_on_objects p' hj₀' fun f => h ((ULiftHomULiftCategory.equiv J).functor.map f)
    /-
      🎉 no goals
    -/


/-- Another induction principle for `IsPreconnected J`:
given a type family `Z : J → Sort*` and
a rule for transporting in *both* directions along a morphism in `J`,
we can transport an `x : Z j₀` to a point in `Z j` for any `j`.
-/
theorem isPreconnected_induction [IsPreconnected J] (Z : J → Sort*)
    (h₁ : ∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), Z j₁ → Z j₂) (h₂ : ∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), Z j₂ → Z j₁)
    {j₀ : J} (x : Z j₀) (j : J) : Nonempty (Z j) :=
  (induct_on_objects { j | Nonempty (Z j) } ⟨x⟩
                    /-
                      J : Type u₁
                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                      inst✝ : CategoryTheory.IsPreconnected J
                      Z : J → Sort u_1
                      h₁ : {j₁ j₂ : J} → Quiver.Hom j₁ j₂ → Z j₁ → Z j₂
                      h₂ : {j₁ j₂ : J} → Quiver.Hom j₁ j₂ → Z j₂ → Z j₁
                      j₀ : J
                      x : Z j₀
                      j j₁✝ j₂✝ : J
                      f : Quiver.Hom j₁✝ j₂✝
                      ⊢ Membership.mem (setOf fun j => Nonempty (Z j)) j₁✝ → Membership.mem (setOf f …
                    -/
                                /-
                                  🎉 no goals
                                -/
      (fun f => ⟨by rintro ⟨y⟩; exact ⟨h₁ f y⟩, by rintro ⟨y⟩; exact ⟨h₂ f y⟩⟩)
                                                               /-
                                                                 🎉 no goals
                                                               -/
      j : _)


/-- If `J` and `K` are equivalent, then if `J` is preconnected then `K` is as well. -/
theorem isPreconnected_of_equivalent {K : Type u₂} [Category.{v₂} K] [IsPreconnected J]
    (e : J ≌ K) : IsPreconnected K where
  iso_constant F k :=
    ⟨calc
        F ≅ e.inverse ⋙ e.functor ⋙ F := (e.invFunIdAssoc F).symm
        _ ≅ e.inverse ⋙ (Functor.const J).obj ((e.functor ⋙ F).obj (e.inverse.obj k)) :=
          isoWhiskerLeft e.inverse (isoConstant (e.functor ⋙ F) (e.inverse.obj k))
        _ ≅ e.inverse ⋙ (Functor.const J).obj (F.obj k) :=
          isoWhiskerLeft _ ((F ⋙ Functor.const J).mapIso (e.counitIso.app k))
                                               /-
                                                 J : Type u₁
                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                 K : Type u₂
                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                 inst✝ : CategoryTheory.IsPreconnected J
                                                 e : CategoryTheory.Equivalence J K
                                                 α✝ : Type u₂
                                                 F : CategoryTheory.Functor K (CategoryTheory.Discrete α✝)
                                                 k : K
                                                 ⊢ ∀ {X Y : K} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((e …
                                               -/
        _ ≅ (Functor.const K).obj (F.obj k) := NatIso.ofComponents fun _ => Iso.refl _⟩
                                               /-
                                                 🎉 no goals
                                               -/


lemma isPreconnected_iff_of_equivalence {K : Type u₂} [Category.{v₂} K] (e : J ≌ K) :
    IsPreconnected J ↔ IsPreconnected K :=
  ⟨fun _ => isPreconnected_of_equivalent e, fun _ => isPreconnected_of_equivalent e.symm⟩


/-- If `J` and `K` are equivalent, then if `J` is connected then `K` is as well. -/
theorem isConnected_of_equivalent {K : Type u₂} [Category.{v₂} K] (e : J ≌ K) [IsConnected J] :
    IsConnected K :=
                                                  /-
                                                    J : Type u₁
                                                    inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                    K : Type u₂
                                                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                    e : CategoryTheory.Equivalence J K
                                                    inst✝ : CategoryTheory.IsConnected J
                                                    ⊢ Nonempty J
                                                  -/
  { is_nonempty := Nonempty.map e.functor.obj (by infer_instance)
                                                  /-
                                                    🎉 no goals
                                                  -/
    toIsPreconnected := isPreconnected_of_equivalent e }


lemma isConnected_iff_of_equivalence {K : Type u₂} [Category.{v₂} K] (e : J ≌ K) :
    IsConnected J ↔ IsConnected K :=
  ⟨fun _ => isConnected_of_equivalent e, fun _ => isConnected_of_equivalent e.symm⟩


/-- If `J` is preconnected, then `Jᵒᵖ` is preconnected as well. -/
instance isPreconnected_op [IsPreconnected J] : IsPreconnected Jᵒᵖ where
  iso_constant := fun {α} F X =>
     /-
       J : Type u₁
       inst✝² : CategoryTheory.Category.{v₁, u₁} J
       K : Type u₂
       inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
       inst✝ : CategoryTheory.IsPreconnected J
       α : Type u₁
       F : CategoryTheory.Functor (Opposite J) (CategoryTheory.Discrete α)
       X : Opposite J
       ⊢ ∀ {X_1 Y : Opposite J} (f : Quiver.Hom X_1 Y), Eq (CategoryTheory.CategorySt …
     -/
    ⟨NatIso.ofComponents fun Y =>
     /-
       🎉 no goals
     -/
      eqToIso (Discrete.ext (Discrete.eq_of_hom ((Nonempty.some
        (IsPreconnected.iso_constant (F.rightOp ⋙ (Discrete.opposite α).functor) (unop X))).app
          (unop Y)).hom))⟩


/-- If `J` is connected, then `Jᵒᵖ` is connected as well. -/
instance isConnected_op [IsConnected J] : IsConnected Jᵒᵖ where
  is_nonempty := Nonempty.intro (op (Classical.arbitrary J))


theorem isPreconnected_of_isPreconnected_op [IsPreconnected Jᵒᵖ] : IsPreconnected J :=
  isPreconnected_of_equivalent (opOpEquivalence J)


theorem isConnected_of_isConnected_op [IsConnected Jᵒᵖ] : IsConnected J :=
  isConnected_of_equivalent (opOpEquivalence J)


variable (J) in
@[simp]
theorem isConnected_op_iff_isConnected : IsConnected Jᵒᵖ ↔ IsConnected J :=
  ⟨fun _ => isConnected_of_isConnected_op, fun _ => isConnected_op⟩


/-- j₁ and j₂ are related by `Zag` if there is a morphism between them. -/
def Zag (j₁ j₂ : J) : Prop :=
  Nonempty (j₁ ⟶ j₂) ∨ Nonempty (j₂ ⟶ j₁)


@[refl] theorem Zag.refl (X : J) : Zag X X := Or.inl ⟨𝟙 _⟩


theorem zag_symmetric : Symmetric (@Zag J _) := fun _ _ h => h.symm


@[symm] theorem Zag.symm {j₁ j₂ : J} (h : Zag j₁ j₂) : Zag j₂ j₁ := zag_symmetric h


theorem Zag.of_hom {j₁ j₂ : J} (f : j₁ ⟶ j₂) : Zag j₁ j₂ := Or.inl ⟨f⟩


theorem Zag.of_inv {j₁ j₂ : J} (f : j₂ ⟶ j₁) : Zag j₁ j₂ := Or.inr ⟨f⟩


/-- `j₁` and `j₂` are related by `Zigzag` if there is a chain of
morphisms from `j₁` to `j₂`, with backward morphisms allowed.
-/
def Zigzag : J → J → Prop :=
  Relation.ReflTransGen Zag


theorem zigzag_symmetric : Symmetric (@Zigzag J _) :=
  Relation.ReflTransGen.symmetric zag_symmetric


theorem zigzag_equivalence : _root_.Equivalence (@Zigzag J _) :=
  _root_.Equivalence.mk Relation.reflexive_reflTransGen (fun h => zigzag_symmetric h)
  (fun h g => Relation.transitive_reflTransGen h g)


@[refl] theorem Zigzag.refl (X : J) : Zigzag X X := zigzag_equivalence.refl _


@[symm] theorem Zigzag.symm {j₁ j₂ : J} (h : Zigzag j₁ j₂) : Zigzag j₂ j₁ := zigzag_symmetric h


@[trans] theorem Zigzag.trans {j₁ j₂ j₃ : J} (h₁ : Zigzag j₁ j₂) (h₂ : Zigzag j₂ j₃) :
    Zigzag j₁ j₃ :=
  zigzag_equivalence.trans h₁ h₂


theorem Zigzag.of_zag {j₁ j₂ : J} (h : Zag j₁ j₂) : Zigzag j₁ j₂ :=
  Relation.ReflTransGen.single h


theorem Zigzag.of_hom {j₁ j₂ : J} (f : j₁ ⟶ j₂) : Zigzag j₁ j₂ :=
  of_zag (Zag.of_hom f)


theorem Zigzag.of_inv {j₁ j₂ : J} (f : j₂ ⟶ j₁) : Zigzag j₁ j₂ :=
  of_zag (Zag.of_inv f)


theorem Zigzag.of_zag_trans {j₁ j₂ j₃ : J} (h₁ : Zag j₁ j₂) (h₂ : Zag j₂ j₃) : Zigzag j₁ j₃ :=
  trans (of_zag h₁) (of_zag h₂)


theorem Zigzag.of_hom_hom {j₁ j₂ j₃ : J} (f₁₂ : j₁ ⟶ j₂) (f₂₃ : j₂ ⟶ j₃) : Zigzag j₁ j₃ :=
  (of_hom f₁₂).trans (of_hom f₂₃)


theorem Zigzag.of_hom_inv {j₁ j₂ j₃ : J} (f₁₂ : j₁ ⟶ j₂) (f₃₂ : j₃ ⟶ j₂) : Zigzag j₁ j₃ :=
  (of_hom f₁₂).trans (of_inv f₃₂)


theorem Zigzag.of_inv_hom {j₁ j₂ j₃ : J} (f₂₁ : j₂ ⟶ j₁) (f₂₃ : j₂ ⟶ j₃) : Zigzag j₁ j₃ :=
  (of_inv f₂₁).trans (of_hom f₂₃)


theorem Zigzag.of_inv_inv {j₁ j₂ j₃ : J} (f₂₁ : j₂ ⟶ j₁) (f₃₂ : j₃ ⟶ j₂) : Zigzag j₁ j₃ :=
  (of_inv f₂₁).trans (of_inv f₃₂)


/-- The setoid given by the equivalence relation `Zigzag`. A quotient for this
setoid is a connected component of the category.
-/
def Zigzag.setoid (J : Type u₂) [Category.{v₁} J] : Setoid J where
  r := Zigzag
  iseqv := zigzag_equivalence


/-- If there is a zigzag from `j₁` to `j₂`, then there is a zigzag from `F j₁` to
`F j₂` as long as `F` is a prefunctor.
-/
theorem zigzag_prefunctor_obj_of_zigzag (F : J ⥤q K) {j₁ j₂ : J} (h : Zigzag j₁ j₂) :
    Zigzag (F.obj j₁) (F.obj j₂) :=
  h.lift _ fun _ _ => Or.imp (Nonempty.map fun f => F.map f) (Nonempty.map fun f => F.map f)


/-- If there is a zigzag from `j₁` to `j₂`, then there is a zigzag from `F j₁` to
`F j₂` as long as `F` is a functor.
-/
theorem zigzag_obj_of_zigzag (F : J ⥤ K) {j₁ j₂ : J} (h : Zigzag j₁ j₂) :
    Zigzag (F.obj j₁) (F.obj j₂) :=
  zigzag_prefunctor_obj_of_zigzag F.toPrefunctor h


/-- A Zag in a discrete category entails an equality of its extremities -/
lemma eq_of_zag (X) {a b : Discrete X} (h : Zag a b) : a.as = b.as :=
  h.elim (fun ⟨f⟩ ↦ Discrete.eq_of_hom f) (fun ⟨f⟩ ↦ (Discrete.eq_of_hom f).symm)


/-- A zigzag in a discrete category entails an equality of its extremities -/
lemma eq_of_zigzag (X) {a b : Discrete X} (h : Zigzag a b) : a.as = b.as := by
  induction h with
  | refl => rfl
  | tail _ h eq  => exact eq.trans (eq_of_zag _ h)

-- TODO: figure out the right way to generalise this to `Zigzag`.

theorem zag_of_zag_obj (F : J ⥤ K) [F.Full] {j₁ j₂ : J} (h : Zag (F.obj j₁) (F.obj j₂)) :
    Zag j₁ j₂ :=
  Or.imp (Nonempty.map F.preimage) (Nonempty.map F.preimage) h


/-- Any equivalence relation containing (⟶) holds for all pairs of a connected category. -/
theorem equiv_relation [IsPreconnected J] (r : J → J → Prop) (hr : _root_.Equivalence r)
    (h : ∀ {j₁ j₂ : J} (_ : j₁ ⟶ j₂), r j₁ j₂) : ∀ j₁ j₂ : J, r j₁ j₂ := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    inst✝ : CategoryTheory.IsPreconnected J
    r : J → J → Prop
    hr : _root_.Equivalence r
    h : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → r j₁ j₂
    ⊢ ∀ (j₁ j₂ : J), r j₁ j₂
  -/
  intros j₁ j₂
  have z : ∀ j : J, r j₁ j :=
    induct_on_objects {k | r j₁ k} (hr.1 j₁)
      fun f => ⟨fun t => hr.3 t (h f), fun t => hr.3 t (hr.2 (h f))⟩
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    inst✝ : CategoryTheory.IsPreconnected J
    r : J → J → Prop
    hr : _root_.Equivalence r
    h : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → r j₁ j₂
    j₁ j₂ : J
    z : ∀ (j : J), r j₁ j
    ⊢ r j₁ j₂
  -/
  exact z j₂
  /-
    🎉 no goals
  -/


/-- In a connected category, any two objects are related by `Zigzag`. -/
theorem isPreconnected_zigzag [IsPreconnected J] (j₁ j₂ : J) : Zigzag j₁ j₂ :=
  equiv_relation _ zigzag_equivalence
    (fun f => Relation.ReflTransGen.single (Or.inl (Nonempty.intro f))) _ _


@[deprecated (since := "2024-02-19")] alias isConnected_zigzag := isPreconnected_zigzag


theorem zigzag_isPreconnected (h : ∀ j₁ j₂ : J, Zigzag j₁ j₂) : IsPreconnected J := by
  /-
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), CategoryTheory.Zigzag j₁ j₂
    ⊢ CategoryTheory.IsPreconnected J
  -/
  apply IsPreconnected.of_constant_of_preserves_morphisms
  /-
    case h
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), CategoryTheory.Zigzag j₁ j₂
    ⊢ ∀ {α : Type u₁} (F : J → α), (∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F …
  -/
  intro α F hF j j'
  /-
    case h
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), CategoryTheory.Zigzag j₁ j₂
    α : Type u₁
    F : J → α
    hF : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
    j j' : J
    ⊢ Eq (F j) (F j')
  -/
  specialize h j j'
  /-
    case h
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    α : Type u₁
    F : J → α
    hF : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
    j j' : J
    h : CategoryTheory.Zigzag j j'
    ⊢ Eq (F j) (F j')
  -/
  induction' h with j₁ j₂ _ hj ih
    /-
      case h.refl
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      α : Type u₁
      F : J → α
      hF : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
      j j' : J
      ⊢ Eq (F j) (F j)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.tail
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      α : Type u₁
      F : J → α
      hF : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
      j j' j₁ j₂ : J
      a✝ : Relation.ReflTransGen CategoryTheory.Zag j j₁
      hj : CategoryTheory.Zag j₁ j₂
      ih : Eq (F j) (F j₁)
      ⊢ Eq (F j) (F j₂)
    -/
  · rw [ih]
    /-
      case h.tail
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      α : Type u₁
      F : J → α
      hF : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
      j j' j₁ j₂ : J
      a✝ : Relation.ReflTransGen CategoryTheory.Zag j j₁
      hj : CategoryTheory.Zag j₁ j₂
      ih : Eq (F j) (F j₁)
      ⊢ Eq (F j₁) (F j₂)
    -/
    rcases hj with (⟨⟨hj⟩⟩|⟨⟨hj⟩⟩)
    /-
      case h.tail.inl.intro
      J : Type u₁
      inst✝ : CategoryTheory.Category.{v₁, u₁} J
      α : Type u₁
      F : J → α
      hF : ∀ {j₁ j₂ : J}, Quiver.Hom j₁ j₂ → Eq (F j₁) (F j₂)
      j j' j₁ j₂ : J
      a✝ : Relation.ReflTransGen CategoryTheory.Zag j j₁
      ih : Eq (F j) (F j₁)
      hj : Quiver.Hom j₁ j₂
      ⊢ Eq (F j₁) (F j₂)
    -/
    exacts [hF hj, (hF hj).symm]
    /-
      🎉 no goals
    -/


/-- If any two objects in a nonempty category are related by `Zigzag`, the category is connected.
-/
theorem zigzag_isConnected [Nonempty J] (h : ∀ j₁ j₂ : J, Zigzag j₁ j₂) : IsConnected J :=
  { zigzag_isPreconnected h with }


theorem exists_zigzag' [IsConnected J] (j₁ j₂ : J) :
    ∃ l, List.Chain Zag j₁ l ∧ List.getLast (j₁ :: l) (List.cons_ne_nil _ _) = j₂ :=
  List.exists_chain_of_relationReflTransGen (isPreconnected_zigzag _ _)


/-- If any two objects in a nonempty category are linked by a sequence of (potentially reversed)
morphisms, then J is connected.

The converse of `exists_zigzag'`.
-/
theorem isPreconnected_of_zigzag (h : ∀ j₁ j₂ : J, ∃ l,
    List.Chain Zag j₁ l ∧ List.getLast (j₁ :: l) (List.cons_ne_nil _ _) = j₂) :
    IsPreconnected J := by
  /-
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), Exists fun l => And (List.Chain CategoryTheory.Zag j₁ l) (E …
    ⊢ CategoryTheory.IsPreconnected J
  -/
  apply zigzag_isPreconnected
  /-
    case h
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), Exists fun l => And (List.Chain CategoryTheory.Zag j₁ l) (E …
    ⊢ ∀ (j₁ j₂ : J), CategoryTheory.Zigzag j₁ j₂
  -/
  intro j₁ j₂
  /-
    case h
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), Exists fun l => And (List.Chain CategoryTheory.Zag j₁ l) (E …
    j₁ j₂ : J
    ⊢ CategoryTheory.Zigzag j₁ j₂
  -/
  rcases h j₁ j₂ with ⟨l, hl₁, hl₂⟩
  /-
    case h.intro.intro
    J : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} J
    h : ∀ (j₁ j₂ : J), Exists fun l => And (List.Chain CategoryTheory.Zag j₁ l) (E …
    j₁ j₂ : J
    l : List J
    hl₁ : List.Chain CategoryTheory.Zag j₁ l
    hl₂ : Eq ((List.cons j₁ l).getLast ⋯) j₂
    ⊢ CategoryTheory.Zigzag j₁ j₂
  -/
  apply List.relationReflTransGen_of_exists_chain l hl₁ hl₂
  /-
    🎉 no goals
  -/


/-- If any two objects in a nonempty category are linked by a sequence of (potentially reversed)
morphisms, then J is connected.

The converse of `exists_zigzag'`.
-/
theorem isConnected_of_zigzag [Nonempty J] (h : ∀ j₁ j₂ : J, ∃ l,
    List.Chain Zag j₁ l ∧ List.getLast (j₁ :: l) (List.cons_ne_nil _ _) = j₂) :
    IsConnected J :=
  { isPreconnected_of_zigzag h with }


/-- If `Discrete α` is connected, then `α` is (type-)equivalent to `PUnit`. -/
def discreteIsConnectedEquivPUnit {α : Type u₁} [IsConnected (Discrete α)] : α ≃ PUnit :=
  Discrete.equivOfEquivalence.{u₁, u₁}
    { functor := Functor.star (Discrete α)
      inverse := Discrete.functor fun _ => Classical.arbitrary _
      unitIso := isoConstant _ (Classical.arbitrary _)
      counitIso := Functor.punitExt _ _ }


/-- For objects `X Y : C`, any natural transformation `α : const X ⟶ const Y` from a connected
category must be constant.
This is the key property of connected categories which we use to establish properties about limits.
-/
theorem nat_trans_from_is_connected [IsPreconnected J] {X Y : C}
    (α : (Functor.const J).obj X ⟶ (Functor.const J).obj Y) :
    ∀ j j' : J, α.app j = (α.app j' : X ⟶ Y) :=
  @constant_of_preserves_morphisms _ _ _ (X ⟶ Y) (fun j => α.app j) fun _ _ f => by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      C : Type w₂
      inst✝¹ : CategoryTheory.Category.{w₁, w₂} C
      inst✝ : CategoryTheory.IsPreconnected J
      X Y : C
      α : Quiver.Hom ((CategoryTheory.Functor.const J).obj X) ((CategoryTheory.Funct …
      x✝¹ x✝ : J
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq ((fun j => α.app j) x✝¹) ((fun j => α.app j) x✝)
    -/
    have := α.naturality f
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      C : Type w₂
      inst✝¹ : CategoryTheory.Category.{w₁, w₂} C
      inst✝ : CategoryTheory.IsPreconnected J
      X Y : C
      α : Quiver.Hom ((CategoryTheory.Functor.const J).obj X) ((CategoryTheory.Funct …
      x✝¹ x✝ : J
      f : Quiver.Hom x✝¹ x✝
      this : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const  …
      ⊢ Eq ((fun j => α.app j) x✝¹) ((fun j => α.app j) x✝)
    -/
    erw [id_comp, comp_id] at this
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      C : Type w₂
      inst✝¹ : CategoryTheory.Category.{w₁, w₂} C
      inst✝ : CategoryTheory.IsPreconnected J
      X Y : C
      α : Quiver.Hom ((CategoryTheory.Functor.const J).obj X) ((CategoryTheory.Funct …
      x✝¹ x✝ : J
      f : Quiver.Hom x✝¹ x✝
      this : Eq (α.app x✝) (α.app x✝¹)
      ⊢ Eq ((fun j => α.app j) x✝¹) ((fun j => α.app j) x✝)
    -/
    exact this.symm
    /-
      🎉 no goals
    -/


instance [IsConnected J] : (Functor.const J : C ⥤ J ⥤ C).Full where
  map_surjective f := ⟨f.app (Classical.arbitrary J), by
    /-
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type w₂
      inst✝¹ : CategoryTheory.Category.{w₁, w₂} C
      inst✝ : CategoryTheory.IsConnected J
      X✝ Y✝ : C
      f : Quiver.Hom ((CategoryTheory.Functor.const J).obj X✝) ((CategoryTheory.Func …
      ⊢ Eq ((CategoryTheory.Functor.const J).map (f.app (Classical.arbitrary J))) f
    -/
    ext j
    /-
      case w.h
      J : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} K
      C : Type w₂
      inst✝¹ : CategoryTheory.Category.{w₁, w₂} C
      inst✝ : CategoryTheory.IsConnected J
      X✝ Y✝ : C
      f : Quiver.Hom ((CategoryTheory.Functor.const J).obj X✝) ((CategoryTheory.Func …
      j : J
      ⊢ Eq (((CategoryTheory.Functor.const J).map (f.app (Classical.arbitrary J))).a …
    -/
    apply nat_trans_from_is_connected f (Classical.arbitrary J) j⟩
    /-
      🎉 no goals
    -/


theorem nonempty_hom_of_preconnected_groupoid {G} [Groupoid G] [IsPreconnected G] :
    ∀ x y : G, Nonempty (x ⟶ y) := by
  /-
    G : Type u_1
    inst✝¹ : CategoryTheory.Groupoid G
    inst✝ : CategoryTheory.IsPreconnected G
    ⊢ ∀ (x y : G), Nonempty (Quiver.Hom x y)
  -/
  refine equiv_relation _ ?_ fun {j₁ j₂} => Nonempty.intro
  exact
    ⟨fun j => ⟨𝟙 _⟩,
     fun {j₁ j₂} => Nonempty.map fun f => inv f,
     fun {_ _ _} => Nonempty.map2 (· ≫ ·)⟩


@[deprecated (since := "2024-02-19")]
alias nonempty_hom_of_connected_groupoid := nonempty_hom_of_preconnected_groupoid


