/-- A cone `t` on `F` is a limit cone if each cone on `F` admits a unique
cone morphism to `t`.

See <https://stacks.math.columbia.edu/tag/002E>.
  -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure IsLimit (t : Cone F) where
  /-- There is a morphism from any cone point to `t.pt` -/
  lift : ∀ s : Cone F, s.pt ⟶ t.pt
  /-- The map makes the triangle with the two natural transformations commute -/
  fac : ∀ (s : Cone F) (j : J), lift s ≫ t.π.app j = s.π.app j := by aesop_cat
  /-- It is the unique such map to do this -/
  uniq : ∀ (s : Cone F) (m : s.pt ⟶ t.pt) (_ : ∀ j : J, m ≫ t.π.app j = s.π.app j), m = lift s := by
    aesop_cat


attribute [reassoc (attr := simp)] IsLimit.fac


instance subsingleton {t : Cone F} : Subsingleton (IsLimit t) :=
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cone F
        ⊢ ∀ (a b : CategoryTheory.Limits.IsLimit t), Eq a b
      -/
  ⟨by intro P Q; cases P; cases Q; congr; aesop_cat⟩
                                          /-
                                            🎉 no goals
                                          -/


/-- Given a natural transformation `α : F ⟶ G`, we give a morphism from the cone point
of any cone over `F` to the cone point of a limit cone over `G`. -/
def map {F G : J ⥤ C} (s : Cone F) {t : Cone G} (P : IsLimit t) (α : F ⟶ G) : s.pt ⟶ t.pt :=
  P.lift ((Cones.postcompose α).obj s)


@[reassoc (attr := simp)]
theorem map_π {F G : J ⥤ C} (c : Cone F) {d : Cone G} (hd : IsLimit d) (α : F ⟶ G) (j : J) :
    hd.map c α ≫ d.π.app j = c.π.app j ≫ α.app j :=
  fac _ _ _


@[simp]
theorem lift_self {c : Cone F} (t : IsLimit c) : t.lift c = 𝟙 c.pt :=
  (t.uniq _ _ fun _ => id_comp _).symm

-- Repackaging the definition in terms of cone morphisms.

/-- The universal morphism from any other cone to a limit cone. -/
@[simps]
def liftConeMorphism {t : Cone F} (h : IsLimit t) (s : Cone F) : s ⟶ t where hom := h.lift s


theorem uniq_cone_morphism {s t : Cone F} (h : IsLimit t) {f f' : s ⟶ t} : f = f' :=
  have : ∀ {g : s ⟶ t}, g = h.liftConeMorphism s := by
    /-
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      s t : CategoryTheory.Limits.Cone F
      h : CategoryTheory.Limits.IsLimit t
      f f' : Quiver.Hom s t
      ⊢ ∀ {g : Quiver.Hom s t}, Eq g (h.liftConeMorphism s)
    -/
    intro g; apply ConeMorphism.ext; exact h.uniq _ _ g.w
                                     /-
                                       🎉 no goals
                                     -/
  this.trans this.symm


/-- Restating the definition of a limit cone in terms of the ∃! operator. -/
theorem existsUnique {t : Cone F} (h : IsLimit t) (s : Cone F) :
    ∃! l : s.pt ⟶ t.pt, ∀ j, l ≫ t.π.app j = s.π.app j :=
  ⟨h.lift s, h.fac s, h.uniq s⟩


/-- Noncomputably make a limit cone from the existence of unique factorizations. -/
def ofExistsUnique {t : Cone F}
    (ht : ∀ s : Cone F, ∃! l : s.pt ⟶ t.pt, ∀ j, l ≫ t.π.app j = s.π.app j) : IsLimit t := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cone F
    ht : ∀ (s : CategoryTheory.Limits.Cone F), ExistsUnique fun l => ∀ (j : J), Eq …
    ⊢ CategoryTheory.Limits.IsLimit t
  -/
  choose s hs hs' using ht
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cone F
    s : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt t.pt
    hs : ∀ (s_1 : CategoryTheory.Limits.Cone F), (fun l => ∀ (j : J), Eq (Category …
    hs' : ∀ (s_1 : CategoryTheory.Limits.Cone F) (y : Quiver.Hom s_1.pt t.pt), (fu …
    ⊢ CategoryTheory.Limits.IsLimit t
  -/
  exact ⟨s, hs, hs'⟩
  /-
    🎉 no goals
  -/


/-- Alternative constructor for `isLimit`,
providing a morphism of cones rather than a morphism between the cone points
and separately the factorisation condition.
-/
@[simps]
def mkConeMorphism {t : Cone F} (lift : ∀ s : Cone F, s ⟶ t)
    (uniq : ∀ (s : Cone F) (m : s ⟶ t), m = lift s) : IsLimit t where
  lift s := (lift s).hom
  uniq s m w :=
                                              /-
                                                J : Type u₁
                                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                K : Type u₂
                                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                C : Type u₃
                                                inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                F : CategoryTheory.Functor J C
                                                t : CategoryTheory.Limits.Cone F
                                                lift : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s t
                                                uniq : ∀ (s : CategoryTheory.Limits.Cone F) (m : Quiver.Hom s t), Eq m (lift s)
                                                s : CategoryTheory.Limits.Cone F
                                                m : Quiver.Hom s.pt t.pt
                                                w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.π.app j)) (s.π.app j)
                                                ⊢ Eq { hom := m, w := w } (lift s)
                                              -/
    have : ConeMorphism.mk m w = lift s := by apply uniq
                                              /-
                                                🎉 no goals
                                              -/
    congrArg ConeMorphism.hom this


/-- Limit cones on `F` are unique up to isomorphism. -/
@[simps]
def uniqueUpToIso {s t : Cone F} (P : IsLimit s) (Q : IsLimit t) : s ≅ t where
  hom := Q.liftConeMorphism s
  inv := P.liftConeMorphism t
  hom_inv_id := P.uniq_cone_morphism
  inv_hom_id := Q.uniq_cone_morphism


/-- Any cone morphism between limit cones is an isomorphism. -/
theorem hom_isIso {s t : Cone F} (P : IsLimit s) (Q : IsLimit t) (f : s ⟶ t) : IsIso f :=
  ⟨⟨P.liftConeMorphism t, ⟨P.uniq_cone_morphism, Q.uniq_cone_morphism⟩⟩⟩


/-- Limits of `F` are unique up to isomorphism. -/
def conePointUniqueUpToIso {s t : Cone F} (P : IsLimit s) (Q : IsLimit t) : s.pt ≅ t.pt :=
  (Cones.forget F).mapIso (uniqueUpToIso P Q)


@[reassoc (attr := simp)]
theorem conePointUniqueUpToIso_hom_comp {s t : Cone F} (P : IsLimit s) (Q : IsLimit t) (j : J) :
    (conePointUniqueUpToIso P Q).hom ≫ t.π.app j = s.π.app j :=
  (uniqueUpToIso P Q).hom.w _


@[reassoc (attr := simp)]
theorem conePointUniqueUpToIso_inv_comp {s t : Cone F} (P : IsLimit s) (Q : IsLimit t) (j : J) :
    (conePointUniqueUpToIso P Q).inv ≫ s.π.app j = t.π.app j :=
  (uniqueUpToIso P Q).inv.w _


@[reassoc (attr := simp)]
theorem lift_comp_conePointUniqueUpToIso_hom {r s t : Cone F} (P : IsLimit s) (Q : IsLimit t) :
    P.lift r ≫ (conePointUniqueUpToIso P Q).hom = Q.lift r :=
                 /-
                   J : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                   C : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   r s t : CategoryTheory.Limits.Cone F
                   P : CategoryTheory.Limits.IsLimit s
                   Q : CategoryTheory.Limits.IsLimit t
                   ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                 -/
  Q.uniq _ _ (by simp)
                 /-
                   🎉 no goals
                 -/


@[reassoc (attr := simp)]
theorem lift_comp_conePointUniqueUpToIso_inv {r s t : Cone F} (P : IsLimit s) (Q : IsLimit t) :
    Q.lift r ≫ (conePointUniqueUpToIso P Q).inv = P.lift r :=
                 /-
                   J : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                   C : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   r s t : CategoryTheory.Limits.Cone F
                   P : CategoryTheory.Limits.IsLimit s
                   Q : CategoryTheory.Limits.IsLimit t
                   ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                 -/
  P.uniq _ _ (by simp)
                 /-
                   🎉 no goals
                 -/


/-- Transport evidence that a cone is a limit cone across an isomorphism of cones. -/
def ofIsoLimit {r t : Cone F} (P : IsLimit r) (i : r ≅ t) : IsLimit t :=
  IsLimit.mkConeMorphism (fun s => P.liftConeMorphism s ≫ i.hom) fun s m => by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit r
      i : CategoryTheory.Iso r t
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s t
      ⊢ Eq m ((fun s => CategoryTheory.CategoryStruct.comp (P.liftConeMorphism s) i. …
    -/
    rw [← i.comp_inv_eq]; apply P.uniq_cone_morphism
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem ofIsoLimit_lift {r t : Cone F} (P : IsLimit r) (i : r ≅ t) (s) :
    (P.ofIsoLimit i).lift s = P.lift s ≫ i.hom.hom :=
  rfl


/-- Isomorphism of cones preserves whether or not they are limiting cones. -/
def equivIsoLimit {r t : Cone F} (i : r ≅ t) : IsLimit r ≃ IsLimit t where
  toFun h := h.ofIsoLimit i
  invFun h := h.ofIsoLimit i.symm
                 /-
                   J : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} J
                   K : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                   C : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   r t : CategoryTheory.Limits.Cone F
                   i : CategoryTheory.Iso r t
                   ⊢ Function.LeftInverse (fun h => h.ofIsoLimit i.symm) fun h => h.ofIsoLimit i
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    J : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} J
                    K : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                    C : Type u₃
                    inst✝ : CategoryTheory.Category.{v₃, u₃} C
                    F : CategoryTheory.Functor J C
                    r t : CategoryTheory.Limits.Cone F
                    i : CategoryTheory.Iso r t
                    ⊢ Function.RightInverse (fun h => h.ofIsoLimit i.symm) fun h => h.ofIsoLimit i
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem equivIsoLimit_apply {r t : Cone F} (i : r ≅ t) (P : IsLimit r) :
    equivIsoLimit i P = P.ofIsoLimit i :=
  rfl


@[simp]
theorem equivIsoLimit_symm_apply {r t : Cone F} (i : r ≅ t) (P : IsLimit t) :
    (equivIsoLimit i).symm P = P.ofIsoLimit i.symm :=
  rfl


/-- If the canonical morphism from a cone point to a limiting cone point is an iso, then the
first cone was limiting also.
-/
def ofPointIso {r t : Cone F} (P : IsLimit r) [i : IsIso (P.lift t)] : IsLimit t :=
  ofIsoLimit P (by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit r
      i : CategoryTheory.IsIso (P.lift t)
      ⊢ CategoryTheory.Iso r t
    -/
    haveI : IsIso (P.liftConeMorphism t).hom := i
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit r
      i : CategoryTheory.IsIso (P.lift t)
      this : CategoryTheory.IsIso (P.liftConeMorphism t).hom
      ⊢ CategoryTheory.Iso r t
    -/
    haveI : IsIso (P.liftConeMorphism t) := Cones.cone_iso_of_hom_iso _
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit r
      i : CategoryTheory.IsIso (P.lift t)
      this✝ : CategoryTheory.IsIso (P.liftConeMorphism t).hom
      this : CategoryTheory.IsIso (P.liftConeMorphism t)
      ⊢ CategoryTheory.Iso r t
    -/
    symm
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cone F
      P : CategoryTheory.Limits.IsLimit r
      i : CategoryTheory.IsIso (P.lift t)
      this✝ : CategoryTheory.IsIso (P.liftConeMorphism t).hom
      this : CategoryTheory.IsIso (P.liftConeMorphism t)
      ⊢ CategoryTheory.Iso t r
    -/
    apply asIso (P.liftConeMorphism t))
    /-
      🎉 no goals
    -/


theorem hom_lift (h : IsLimit t) {W : C} (m : W ⟶ t.pt) :
    m = h.lift { pt := W, π := { app := fun b => m ≫ t.π.app b } } :=
  h.uniq { pt := W, π := { app := fun b => m ≫ t.π.app b } } m fun _ => rfl


/-- Two morphisms into a limit are equal if their compositions with
  each cone morphism are equal. -/
theorem hom_ext (h : IsLimit t) {W : C} {f f' : W ⟶ t.pt}
    (w : ∀ j, f ≫ t.π.app j = f' ≫ t.π.app j) :
    f = f' := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cone F
    h : CategoryTheory.Limits.IsLimit t
    W : C
    f f' : Quiver.Hom W t.pt
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp f (t.π.app j)) (Category …
    ⊢ Eq f f'
  -/
  rw [h.hom_lift f, h.hom_lift f']; congr; exact funext w
                                           /-
                                             🎉 no goals
                                           -/


/-- Given a right adjoint functor between categories of cones,
the image of a limit cone is a limit cone.
-/
def ofRightAdjoint {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D} {left : Cone F ⥤ Cone G}
    {right : Cone G ⥤ Cone F}
    (adj : left ⊣ right) {c : Cone G} (t : IsLimit c) : IsLimit (right.obj c) :=
  mkConeMorphism (fun s => adj.homEquiv s c (t.liftConeMorphism _))
    fun _ _ => (Adjunction.eq_homEquiv_apply _ _ _).2 t.uniq_cone_morphism


/-- Given two functors which have equivalent categories of cones, we can transport a limiting cone
across the equivalence.
-/
def ofConeEquiv {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D} (h : Cone G ≌ Cone F) {c : Cone G} :
    IsLimit (h.functor.obj c) ≃ IsLimit c where
  toFun P := ofIsoLimit (ofRightAdjoint h.toAdjunction P) (h.unitIso.symm.app c)
  invFun := ofRightAdjoint h.symm.toAdjunction
                 /-
                   J : Type u₁
                   inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                   K : Type u₂
                   inst✝² : CategoryTheory.Category.{v₂, u₂} K
                   C : Type u₃
                   inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   t : CategoryTheory.Limits.Cone F
                   D : Type u₄
                   inst✝ : CategoryTheory.Category.{v₄, u₄} D
                   G : CategoryTheory.Functor K D
                   h : CategoryTheory.Equivalence (CategoryTheory.Limits.Cone G) (CategoryTheory. …
                   c : CategoryTheory.Limits.Cone G
                   ⊢ Function.LeftInverse (CategoryTheory.Limits.IsLimit.ofRightAdjoint h.symm.to …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    J : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                    K : Type u₂
                    inst✝² : CategoryTheory.Category.{v₂, u₂} K
                    C : Type u₃
                    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                    F : CategoryTheory.Functor J C
                    t : CategoryTheory.Limits.Cone F
                    D : Type u₄
                    inst✝ : CategoryTheory.Category.{v₄, u₄} D
                    G : CategoryTheory.Functor K D
                    h : CategoryTheory.Equivalence (CategoryTheory.Limits.Cone G) (CategoryTheory. …
                    c : CategoryTheory.Limits.Cone G
                    ⊢ Function.RightInverse (CategoryTheory.Limits.IsLimit.ofRightAdjoint h.symm.t …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem ofConeEquiv_apply_desc {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D} (h : Cone G ≌ Cone F)
    {c : Cone G} (P : IsLimit (h.functor.obj c)) (s) :
    (ofConeEquiv h P).lift s =
      ((h.unitIso.hom.app s).hom ≫ (h.inverse.map (P.liftConeMorphism (h.functor.obj s))).hom) ≫
        (h.unitIso.inv.app c).hom :=
  rfl


@[simp]
theorem ofConeEquiv_symm_apply_desc {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D}
    (h : Cone G ≌ Cone F) {c : Cone G} (P : IsLimit c) (s) :
    ((ofConeEquiv h).symm P).lift s =
      (h.counitIso.inv.app s).hom ≫ (h.functor.map (P.liftConeMorphism (h.inverse.obj s))).hom :=
  rfl


/--
A cone postcomposed with a natural isomorphism is a limit cone if and only if the original cone is.
-/
def postcomposeHomEquiv {F G : J ⥤ C} (α : F ≅ G) (c : Cone F) :
    IsLimit ((Cones.postcompose α.hom).obj c) ≃ IsLimit c :=
  ofConeEquiv (Cones.postcomposeEquivalence α)


/-- A cone postcomposed with the inverse of a natural isomorphism is a limit cone if and only if
the original cone is.
-/
def postcomposeInvEquiv {F G : J ⥤ C} (α : F ≅ G) (c : Cone G) :
    IsLimit ((Cones.postcompose α.inv).obj c) ≃ IsLimit c :=
  postcomposeHomEquiv α.symm c


/-- Constructing an equivalence `IsLimit c ≃ IsLimit d` from a natural isomorphism
between the underlying functors, and then an isomorphism between `c` transported along this and `d`.
-/
def equivOfNatIsoOfIso {F G : J ⥤ C} (α : F ≅ G) (c : Cone F) (d : Cone G)
    (w : (Cones.postcompose α.hom).obj c ≅ d) : IsLimit c ≃ IsLimit d :=
  (postcomposeHomEquiv α _).symm.trans (equivIsoLimit w)


/-- The cone points of two limit cones for naturally isomorphic functors
are themselves isomorphic.
-/
@[simps]
def conePointsIsoOfNatIso {F G : J ⥤ C} {s : Cone F} {t : Cone G} (P : IsLimit s) (Q : IsLimit t)
    (w : F ≅ G) : s.pt ≅ t.pt where
  hom := Q.map s w.hom
  inv := P.map t w.inv
                              /-
                                J : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                K : Type u₂
                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                C : Type u₃
                                inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                F✝ : CategoryTheory.Functor J C
                                t✝ : CategoryTheory.Limits.Cone F✝
                                F G : CategoryTheory.Functor J C
                                s : CategoryTheory.Limits.Cone F
                                t : CategoryTheory.Limits.Cone G
                                P : CategoryTheory.Limits.IsLimit s
                                Q : CategoryTheory.Limits.IsLimit t
                                w : CategoryTheory.Iso F G
                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                              -/
  hom_inv_id := P.hom_ext (by aesop_cat)
                              /-
                                🎉 no goals
                              -/
                              /-
                                J : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                K : Type u₂
                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                C : Type u₃
                                inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                F✝ : CategoryTheory.Functor J C
                                t✝ : CategoryTheory.Limits.Cone F✝
                                F G : CategoryTheory.Functor J C
                                s : CategoryTheory.Limits.Cone F
                                t : CategoryTheory.Limits.Cone G
                                P : CategoryTheory.Limits.IsLimit s
                                Q : CategoryTheory.Limits.IsLimit t
                                w : CategoryTheory.Iso F G
                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                              -/
  inv_hom_id := Q.hom_ext (by aesop_cat)
                              /-
                                🎉 no goals
                              -/


@[reassoc]
theorem conePointsIsoOfNatIso_hom_comp {F G : J ⥤ C} {s : Cone F} {t : Cone G} (P : IsLimit s)
    (Q : IsLimit t) (w : F ≅ G) (j : J) :
                                                                                  /-
                                                                                    J : Type u₁
                                                                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                                    C : Type u₃
                                                                                    inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                                    F G : CategoryTheory.Functor J C
                                                                                    s : CategoryTheory.Limits.Cone F
                                                                                    t : CategoryTheory.Limits.Cone G
                                                                                    P : CategoryTheory.Limits.IsLimit s
                                                                                    Q : CategoryTheory.Limits.IsLimit t
                                                                                    w : CategoryTheory.Iso F G
                                                                                    j : J
                                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.conePointsIsoOfNatIso Q w).hom (t. …
                                                                                  -/
    (conePointsIsoOfNatIso P Q w).hom ≫ t.π.app j = s.π.app j ≫ w.hom.app j := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[reassoc]
theorem conePointsIsoOfNatIso_inv_comp {F G : J ⥤ C} {s : Cone F} {t : Cone G} (P : IsLimit s)
    (Q : IsLimit t) (w : F ≅ G) (j : J) :
                                                                                  /-
                                                                                    J : Type u₁
                                                                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                                    C : Type u₃
                                                                                    inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                                    F G : CategoryTheory.Functor J C
                                                                                    s : CategoryTheory.Limits.Cone F
                                                                                    t : CategoryTheory.Limits.Cone G
                                                                                    P : CategoryTheory.Limits.IsLimit s
                                                                                    Q : CategoryTheory.Limits.IsLimit t
                                                                                    w : CategoryTheory.Iso F G
                                                                                    j : J
                                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.conePointsIsoOfNatIso Q w).inv (s. …
                                                                                  -/
    (conePointsIsoOfNatIso P Q w).inv ≫ s.π.app j = t.π.app j ≫ w.inv.app j := by simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[reassoc]
theorem lift_comp_conePointsIsoOfNatIso_hom {F G : J ⥤ C} {r s : Cone F} {t : Cone G}
    (P : IsLimit s) (Q : IsLimit t) (w : F ≅ G) :
    P.lift r ≫ (conePointsIsoOfNatIso P Q w).hom = Q.map r w.hom :=
                /-
                  J : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                  C : Type u₃
                  inst✝ : CategoryTheory.Category.{v₃, u₃} C
                  F G : CategoryTheory.Functor J C
                  r s : CategoryTheory.Limits.Cone F
                  t : CategoryTheory.Limits.Cone G
                  P : CategoryTheory.Limits.IsLimit s
                  Q : CategoryTheory.Limits.IsLimit t
                  w : CategoryTheory.Iso F G
                  ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                -/
  Q.hom_ext (by simp)
                /-
                  🎉 no goals
                -/


@[reassoc]
theorem lift_comp_conePointsIsoOfNatIso_inv {F G : J ⥤ C} {r s : Cone G} {t : Cone F}
    (P : IsLimit t) (Q : IsLimit s) (w : F ≅ G) :
    Q.lift r ≫ (conePointsIsoOfNatIso P Q w).inv = P.map r w.inv :=
                /-
                  J : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                  C : Type u₃
                  inst✝ : CategoryTheory.Category.{v₃, u₃} C
                  F G : CategoryTheory.Functor J C
                  r s : CategoryTheory.Limits.Cone G
                  t : CategoryTheory.Limits.Cone F
                  P : CategoryTheory.Limits.IsLimit t
                  Q : CategoryTheory.Limits.IsLimit s
                  w : CategoryTheory.Iso F G
                  ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                -/
  P.hom_ext (by simp)
                /-
                  🎉 no goals
                -/


/-- If `s : Cone F` is a limit cone, so is `s` whiskered by an equivalence `e`.
-/
def whiskerEquivalence {s : Cone F} (P : IsLimit s) (e : K ≌ J) : IsLimit (s.whisker e.functor) :=
  ofRightAdjoint (Cones.whiskeringEquivalence e).symm.toAdjunction P


/-- If `s : Cone F` whiskered by an equivalence `e` is a limit cone, so is `s`.
-/
def ofWhiskerEquivalence {s : Cone F} (e : K ≌ J) (P : IsLimit (s.whisker e.functor)) : IsLimit s :=
  equivIsoLimit ((Cones.whiskeringEquivalence e).unitIso.app s).symm
    (ofRightAdjoint (Cones.whiskeringEquivalence e).toAdjunction P)


/-- Given an equivalence of diagrams `e`, `s` is a limit cone iff `s.whisker e.functor` is.
-/
def whiskerEquivalenceEquiv {s : Cone F} (e : K ≌ J) : IsLimit s ≃ IsLimit (s.whisker e.functor) :=
                                                               /-
                                                                 J : Type u₁
                                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                 K : Type u₂
                                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                                 C : Type u₃
                                                                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                 F : CategoryTheory.Functor J C
                                                                 t s : CategoryTheory.Limits.Cone F
                                                                 e : CategoryTheory.Equivalence K J
                                                                 ⊢ Function.LeftInverse (CategoryTheory.Limits.IsLimit.ofWhiskerEquivalence e)  …
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  ⟨fun h => h.whiskerEquivalence e, ofWhiskerEquivalence e, by aesop_cat, by aesop_cat⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- A limit cone extended by an isomorphism is a limit cone. -/
def extendIso {s : Cone F} {X : C} (i : X ⟶ s.pt) [IsIso i] (hs : IsLimit s) :
    IsLimit (s.extend i) :=
  IsLimit.ofIsoLimit hs (Cones.extendIso s (asIso i)).symm


/-- A cone is a limit cone if its extension by an isomorphism is. -/
def ofExtendIso {s : Cone F} {X : C} (i : X ⟶ s.pt) [IsIso i] (hs : IsLimit (s.extend i)) :
    IsLimit s :=
  IsLimit.ofIsoLimit hs (Cones.extendIso s (asIso i))


/-- A cone is a limit cone iff its extension by an isomorphism is. -/
def extendIsoEquiv {s : Cone F} {X : C} (i : X ⟶ s.pt) [IsIso i] :
    IsLimit s ≃ IsLimit (s.extend i) :=
  equivOfSubsingletonOfSubsingleton (extendIso i) (ofExtendIso i)


/-- We can prove two cone points `(s : Cone F).pt` and `(t : Cone G).pt` are isomorphic if
* both cones are limit cones
* their indexing categories are equivalent via some `e : J ≌ K`,
* the triangle of functors commutes up to a natural isomorphism: `e.functor ⋙ G ≅ F`.

This is the most general form of uniqueness of cone points,
allowing relabelling of both the indexing category (up to equivalence)
and the functor (up to natural isomorphism).
-/
@[simps]
def conePointsIsoOfEquivalence {F : J ⥤ C} {s : Cone F} {G : K ⥤ C} {t : Cone G} (P : IsLimit s)
    (Q : IsLimit t) (e : J ≌ K) (w : e.functor ⋙ G ≅ F) : s.pt ≅ t.pt :=
  let w' : e.inverse ⋙ F ≅ G := (isoWhiskerLeft e.inverse w).symm ≪≫ invFunIdAssoc e G
  { hom := Q.lift ((Cones.equivalenceOfReindexing e.symm w').functor.obj s)
    inv := P.lift ((Cones.equivalenceOfReindexing e w).functor.obj t)
    hom_inv_id := by
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Q.lift ((CategoryTheory.Limits.Cones …
      -/
      apply hom_ext P; intro j
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      dsimp [w']
      simp only [Limits.Cone.whisker_π, Limits.Cones.postcompose_obj_π, fac, whiskerLeft_app,
        assoc, id_comp, invFunIdAssoc_hom_app, fac_assoc, NatTrans.comp_app]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app (e.inverse.obj (e.functor.ob …
      -/
      rw [counit_app_functor, ← Functor.comp_map]
      have l :
        NatTrans.app w.hom j = NatTrans.app w.hom (Prefunctor.obj (𝟭 J).toPrefunctor j) := by dsimp
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        l : Eq (w.hom.app j) (w.hom.app ((CategoryTheory.Functor.id J).obj j))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app (e.inverse.obj (e.functor.ob …
      -/
      rw [l,w.hom.naturality]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        l : Eq (w.hom.app j) (w.hom.app ((CategoryTheory.Functor.id J).obj j))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.π.app (e.inverse.obj (e.functor.ob …
      -/
      simp
      /-
        🎉 no goals
      -/
    inv_hom_id := by
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.lift ((CategoryTheory.Limits.Cones …
      -/
      apply hom_ext Q
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cone G
        P : CategoryTheory.Limits.IsLimit s
        Q : CategoryTheory.Limits.IsLimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
      -/
      aesop_cat }
      /-
        🎉 no goals
      -/


/-- The universal property of a limit cone: a map `W ⟶ X` is the same as
  a cone on `F` with cone point `W`. -/
def homIso (h : IsLimit t) (W : C) : ULift.{u₁} (W ⟶ t.pt : Type v₃) ≅ (const J).obj W ⟶ F where
  hom f := (t.extend f.down).π
  inv π := ⟨h.lift { pt := W, π }⟩
  hom_inv_id := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      h : CategoryTheory.Limits.IsLimit t
      W : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => (t.extend f.down).π) fun π  …
    -/
    funext f; apply ULift.ext
    /-
      case h.h
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      h : CategoryTheory.Limits.IsLimit t
      W : C
      f : ULift.{u₁, v₃} (Quiver.Hom W t.pt)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun f => (t.extend f.down).π) (fun π …
    -/
    apply h.hom_ext; intro j; simp
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem homIso_hom (h : IsLimit t) {W : C} (f : ULift.{u₁} (W ⟶ t.pt)) :
    (IsLimit.homIso h W).hom f = (t.extend f.down).π :=
  rfl


/-- The limit of `F` represents the functor taking `W` to
  the set of cones on `F` with cone point `W`. -/
def natIso (h : IsLimit t) : yoneda.obj t.pt ⋙ uliftFunctor.{u₁} ≅ F.cones :=
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cone F
    h : CategoryTheory.Limits.IsLimit t
    ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
  -/
  NatIso.ofComponents fun W => IsLimit.homIso h (unop W)
  /-
    🎉 no goals
  -/


/-- Another, more explicit, formulation of the universal property of a limit cone.
See also `homIso`. -/
def homIso' (h : IsLimit t) (W : C) :
    ULift.{u₁} (W ⟶ t.pt : Type v₃) ≅
      { p : ∀ j, W ⟶ F.obj j // ∀ {j j'} (f : j ⟶ j'), p j ≫ F.map f = p j' } :=
  h.homIso W ≪≫
    { hom := fun π =>
                                       /-
                                         J : Type u₁
                                         inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                         K : Type u₂
                                         inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                         C : Type u₃
                                         inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                         F : CategoryTheory.Functor J C
                                         t : CategoryTheory.Limits.Cone F
                                         h : CategoryTheory.Limits.IsLimit t
                                         W : C
                                         π : Quiver.Hom ((CategoryTheory.Functor.const J).obj W) F
                                         j✝ j'✝ : J
                                         f : Quiver.Hom j✝ j'✝
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun j => π.app j) j✝) (F.map f)) (( …
                                       -/
        ⟨fun j => π.app j, fun f => by convert ← (π.naturality f).symm; apply id_comp⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      inv := fun p =>
        { app := fun j => p.1 j
                                         /-
                                           J : Type u₁
                                           inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                           K : Type u₂
                                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                           C : Type u₃
                                           inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                           F : CategoryTheory.Functor J C
                                           t : CategoryTheory.Limits.Cone F
                                           h : CategoryTheory.Limits.IsLimit t
                                           W : C
                                           p : Subtype fun p => ∀ {j j' : J} (f : Quiver.Hom j j'), Eq (CategoryTheory.Ca …
                                           j j' : J
                                           f : Quiver.Hom j j'
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
                                         -/
          naturality := fun j j' f => by dsimp; rw [id_comp]; exact (p.2 f).symm } }
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If G : C → D is a faithful functor which sends t to a limit cone,
  then it suffices to check that the induced maps for the image of t
  can be lifted to maps of C. -/
def ofFaithful {t : Cone F} {D : Type u₄} [Category.{v₄} D] (G : C ⥤ D) [G.Faithful]
    (ht : IsLimit (mapCone G t)) (lift : ∀ s : Cone F, s.pt ⟶ t.pt)
    (h : ∀ s, G.map (lift s) = ht.lift (mapCone G s)) : IsLimit t :=
  { lift
                         /-
                           J : Type u₁
                           inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                           K : Type u₂
                           inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                           C : Type u₃
                           inst✝² : CategoryTheory.Category.{v₃, u₃} C
                           F : CategoryTheory.Functor J C
                           t✝ t : CategoryTheory.Limits.Cone F
                           D : Type u₄
                           inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                           G : CategoryTheory.Functor C D
                           inst✝ : G.Faithful
                           ht : CategoryTheory.Limits.IsLimit (G.mapCone t)
                           lift : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt t.pt
                           h : ∀ (s : CategoryTheory.Limits.Cone F), Eq (G.map (lift s)) (ht.lift (G.mapC …
                           s : CategoryTheory.Limits.Cone F
                           j : J
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app j)) (s.π.app j)
                         -/
    fac := fun s j => by apply G.map_injective; rw [G.map_comp, h]; apply ht.fac
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    uniq := fun s m w => by
      /-
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsLimit (G.mapCone t)
        lift : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt t.pt
        h : ∀ (s : CategoryTheory.Limits.Cone F), Eq (G.map (lift s)) (ht.lift (G.mapC …
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.π.app j)) (s.π.app j)
        ⊢ Eq m (lift s)
      -/
      apply G.map_injective; rw [h]
      /-
        case a
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsLimit (G.mapCone t)
        lift : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt t.pt
        h : ∀ (s : CategoryTheory.Limits.Cone F), Eq (G.map (lift s)) (ht.lift (G.mapC …
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.π.app j)) (s.π.app j)
        ⊢ Eq (G.map m) (ht.lift (G.mapCone s))
      -/
      refine ht.uniq (mapCone G s) _ fun j => ?_
      /-
        case a
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsLimit (G.mapCone t)
        lift : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt t.pt
        h : ∀ (s : CategoryTheory.Limits.Cone F), Eq (G.map (lift s)) (ht.lift (G.mapC …
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.π.app j)) (s.π.app j)
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map m) ((G.mapCone t).π.app j)) (( …
      -/
      convert ← congrArg (fun f => G.map f) (w j)
      /-
        case h.e'_2.h
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsLimit (G.mapCone t)
        lift : (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt t.pt
        h : ∀ (s : CategoryTheory.Limits.Cone F), Eq (G.map (lift s)) (ht.lift (G.mapC …
        s : CategoryTheory.Limits.Cone F
        m : Quiver.Hom s.pt t.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m (t.π.app j)) (s.π.app j)
        j : J
        e_1✝ : Eq (Quiver.Hom (G.obj s.pt) (G.obj (F.obj j))) (Quiver.Hom (G.mapCone s …
        ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp m (t.π.app j))) (CategoryTheor …
      -/
      apply G.map_comp }
      /-
        🎉 no goals
      -/


/-- If `F` and `G` are naturally isomorphic, then `F.mapCone c` being a limit implies
`G.mapCone c` is also a limit.
-/
def mapConeEquiv {D : Type u₄} [Category.{v₄} D] {K : J ⥤ C} {F G : C ⥤ D} (h : F ≅ G) {c : Cone K}
    (t : IsLimit (mapCone F c)) : IsLimit (mapCone G c) := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    F✝ : CategoryTheory.Functor J C
    t✝ : CategoryTheory.Limits.Cone F✝
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    K : CategoryTheory.Functor J C
    F G : CategoryTheory.Functor C D
    h : CategoryTheory.Iso F G
    c : CategoryTheory.Limits.Cone K
    t : CategoryTheory.Limits.IsLimit (F.mapCone c)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone c)
  -/
  apply postcomposeInvEquiv (isoWhiskerLeft K h : _) (mapCone G c) _
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    F✝ : CategoryTheory.Functor J C
    t✝ : CategoryTheory.Limits.Cone F✝
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    K : CategoryTheory.Functor J C
    F G : CategoryTheory.Functor C D
    h : CategoryTheory.Iso F G
    c : CategoryTheory.Limits.Cone K
    t : CategoryTheory.Limits.IsLimit (F.mapCone c)
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose (Cat …
  -/
  apply t.ofIsoLimit (postcomposeWhiskerLeftMapCone h.symm c).symm
  /-
    🎉 no goals
  -/


/-- A cone is a limit cone exactly if
there is a unique cone morphism from any other cone.
-/
def isoUniqueConeMorphism {t : Cone F} : IsLimit t ≅ ∀ s, Unique (s ⟶ t) where
  hom h s :=
    { default := h.liftConeMorphism s
      uniq := fun _ => h.uniq_cone_morphism }
  inv h :=
    { lift := fun s => (h s).default.hom
      uniq := fun s f w => congrArg ConeMorphism.hom ((h s).uniq ⟨f, w⟩) }


/-- If `F.cones` is represented by `X`, each morphism `f : Y ⟶ X` gives a cone with cone point
`Y`. -/
def coneOfHom {Y : C} (f : Y ⟶ X) : Cone F where
  pt := Y
  π := h.hom.app (op Y) ⟨f⟩


/-- If `F.cones` is represented by `X`, each cone `s` gives a morphism `s.pt ⟶ X`. -/
def homOfCone (s : Cone F) : s.pt ⟶ X :=
  (h.inv.app (op s.pt) s.π).down


@[simp]
theorem coneOfHom_homOfCone (s : Cone F) : coneOfHom h (homOfCone h s) = s := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    s : CategoryTheory.Limits.Cone F
    ⊢ Eq (CategoryTheory.Limits.IsLimit.OfNatIso.coneOfHom h (CategoryTheory.Limit …
  -/
  dsimp [coneOfHom, homOfCone]
  match s with
  | .mk s_pt s_π =>
    congr; dsimp
    convert congrFun (congrFun (congrArg NatTrans.app h.inv_hom_id) (op s_pt)) s_π using 1


@[simp]
theorem homOfCone_coneOfHom {Y : C} (f : Y ⟶ X) : homOfCone h (coneOfHom h f) = f :=
  congrArg ULift.down (congrFun (congrFun (congrArg NatTrans.app h.hom_inv_id) (op Y)) ⟨f⟩ : _)


/-- If `F.cones` is represented by `X`, the cone corresponding to the identity morphism on `X`
will be a limit cone. -/
def limitCone : Cone F :=
  coneOfHom h (𝟙 X)


/-- If `F.cones` is represented by `X`, the cone corresponding to a morphism `f : Y ⟶ X` is
the limit cone extended by `f`. -/
theorem coneOfHom_fac {Y : C} (f : Y ⟶ X) : coneOfHom h f = (limitCone h).extend f := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    ⊢ Eq (CategoryTheory.Limits.IsLimit.OfNatIso.coneOfHom h f) ((CategoryTheory.L …
  -/
  dsimp [coneOfHom, limitCone, Cone.extend]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    ⊢ Eq { pt := Y, π := h.hom.app { unop := Y } { down := f } } { pt := Y, π := C …
  -/
  congr with j
  /-
    case e_π.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    j : J
    ⊢ Eq ((h.hom.app { unop := Y } { down := f }).app j) ((CategoryTheory.Category …
  -/
  have t := congrFun (h.hom.naturality f.op) ⟨𝟙 X⟩
  /-
    case e_π.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    j : J
    t : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.yoneda.obj X).com …
    ⊢ Eq ((h.hom.app { unop := Y } { down := f }).app j) ((CategoryTheory.Category …
  -/
  dsimp at t
  /-
    case e_π.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    j : J
    t : Eq (h.hom.app { unop := Y } { down := CategoryTheory.CategoryStruct.comp f …
    ⊢ Eq ((h.hom.app { unop := Y } { down := f }).app j) ((CategoryTheory.Category …
  -/
  simp only [comp_id] at t
  /-
    case e_π.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    j : J
    t : Eq (h.hom.app { unop := Y } { down := f }) (F.cones.map f.op (h.hom.app {  …
    ⊢ Eq ((h.hom.app { unop := Y } { down := f }).app j) ((CategoryTheory.Category …
  -/
  rw [congrFun (congrArg NatTrans.app t) j]
  /-
    case e_π.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    Y : C
    f : Quiver.Hom Y X
    j : J
    t : Eq (h.hom.app { unop := Y } { down := f }) (F.cones.map f.op (h.hom.app {  …
    ⊢ Eq ((F.cones.map f.op (h.hom.app { unop := X } { down := CategoryTheory.Cate …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `F.cones` is represented by `X`, any cone is the extension of the limit cone by the
corresponding morphism. -/
theorem cone_fac (s : Cone F) : (limitCone h).extend (homOfCone h s) = s := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    s : CategoryTheory.Limits.Cone F
    ⊢ Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extend (CategoryThe …
  -/
  rw [← coneOfHom_homOfCone h s]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    s : CategoryTheory.Limits.Cone F
    ⊢ Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extend (CategoryThe …
  -/
  conv_lhs => simp only [homOfCone_coneOfHom]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
    s : CategoryTheory.Limits.Cone F
    ⊢ Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extend (CategoryThe …
  -/
  apply (coneOfHom_fac _ _).symm
  /-
    🎉 no goals
  -/


/-- If `F.cones` is representable, then the cone corresponding to the identity morphism on
the representing object is a limit cone.
-/
def ofNatIso {X : C} (h : yoneda.obj X ⋙ uliftFunctor.{u₁} ≅ F.cones) : IsLimit (limitCone h) where
  lift s := homOfCone h s
  fac s j := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.IsLi …
    -/
    have h := cone_fac h s
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h✝ : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uli …
      s : CategoryTheory.Limits.Cone F
      j : J
      h : Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h✝).extend (Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.IsLi …
    -/
    cases s
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h✝ : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.uli …
      j : J
      pt✝ : C
      π✝ : Quiver.Hom ((CategoryTheory.Functor.const J).obj pt✝) F
      h : Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h✝).extend (Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.IsLi …
    -/
    injection h with h₁ h₂
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      j : J
      pt✝ : C
      π✝ : Quiver.Hom ((CategoryTheory.Functor.const J).obj pt✝) F
      h₁ : Eq { pt := pt✝, π := π✝ }.pt pt✝
      h₂ : Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extensions.app { …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.IsLi …
    -/
    simp only [heq_iff_eq] at h₂
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      j : J
      pt✝ : C
      π✝ : Quiver.Hom ((CategoryTheory.Functor.const J).obj pt✝) F
      h₁ : Eq { pt := pt✝, π := π✝ }.pt pt✝
      h₂ : Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extensions.app { …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.IsLi …
    -/
    conv_rhs => rw [← h₂]
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      j : J
      pt✝ : C
      π✝ : Quiver.Hom ((CategoryTheory.Functor.const J).obj pt✝) F
      h₁ : Eq { pt := pt✝, π := π✝ }.pt pt✝
      h₂ : Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extensions.app { …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.IsLi …
    -/
    rfl
    /-
      🎉 no goals
    -/
  uniq s m w := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq m ((fun s => CategoryTheory.Limits.IsLimit.OfNatIso.homOfCone h s) s)
    -/
    rw [← homOfCone_coneOfHom h m]
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq (CategoryTheory.Limits.IsLimit.OfNatIso.homOfCone h (CategoryTheory.Limit …
    -/
    congr
    /-
      case h.e_8.h
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq (CategoryTheory.Limits.IsLimit.OfNatIso.coneOfHom h m) s
    -/
    rw [coneOfHom_fac]
    /-
      case h.e_8.h
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.yoneda.obj X).comp CategoryTheory.ulif …
      s : CategoryTheory.Limits.Cone F
      m : Quiver.Hom s.pt (CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).pt
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Limit …
      ⊢ Eq ((CategoryTheory.Limits.IsLimit.OfNatIso.limitCone h).extend m) s
    -/
    dsimp [Cone.extend]; cases s; congr with j; exact w j
                                                /-
                                                  🎉 no goals
                                                -/


/-- A cocone `t` on `F` is a colimit cocone if each cocone on `F` admits a unique
cocone morphism from `t`.

See <https://stacks.math.columbia.edu/tag/002F>.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure IsColimit (t : Cocone F) where
  /-- `t.pt` maps to all other cocone covertices -/
  desc : ∀ s : Cocone F, t.pt ⟶ s.pt
  /-- The map `desc` makes the diagram with the natural transformations commute -/
  fac : ∀ (s : Cocone F) (j : J), t.ι.app j ≫ desc s = s.ι.app j := by aesop_cat
  /-- `desc` is the unique such map -/
  uniq :
    ∀ (s : Cocone F) (m : t.pt ⟶ s.pt) (_ : ∀ j : J, t.ι.app j ≫ m = s.ι.app j), m = desc s := by
    aesop_cat


attribute [reassoc (attr := simp)] IsColimit.fac


instance subsingleton {t : Cocone F} : Subsingleton (IsColimit t) :=
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t : CategoryTheory.Limits.Cocone F
        ⊢ ∀ (a b : CategoryTheory.Limits.IsColimit t), Eq a b
      -/
  ⟨by intro P Q; cases P; cases Q; congr; aesop_cat⟩
                                          /-
                                            🎉 no goals
                                          -/


/-- Given a natural transformation `α : F ⟶ G`, we give a morphism from the cocone point
of a colimit cocone over `F` to the cocone point of any cocone over `G`. -/
def map {F G : J ⥤ C} {s : Cocone F} (P : IsColimit s) (t : Cocone G) (α : F ⟶ G) : s.pt ⟶ t.pt :=
  P.desc ((Cocones.precompose α).obj t)


@[reassoc (attr := simp)]
theorem ι_map {F G : J ⥤ C} {c : Cocone F} (hc : IsColimit c) (d : Cocone G) (α : F ⟶ G) (j : J) :
    c.ι.app j ≫ IsColimit.map hc d α = α.app j ≫ d.ι.app j :=
  fac _ _ _


@[simp]
theorem desc_self {t : Cocone F} (h : IsColimit t) : h.desc t = 𝟙 t.pt :=
  (h.uniq _ _ fun _ => comp_id _).symm

-- Repackaging the definition in terms of cocone morphisms.

/-- The universal morphism from a colimit cocone to any other cocone. -/
@[simps]
def descCoconeMorphism {t : Cocone F} (h : IsColimit t) (s : Cocone F) : t ⟶ s where hom := h.desc s


theorem uniq_cocone_morphism {s t : Cocone F} (h : IsColimit t) {f f' : t ⟶ s} : f = f' :=
  have : ∀ {g : t ⟶ s}, g = h.descCoconeMorphism s := by
    /-
      J : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      s t : CategoryTheory.Limits.Cocone F
      h : CategoryTheory.Limits.IsColimit t
      f f' : Quiver.Hom t s
      ⊢ ∀ {g : Quiver.Hom t s}, Eq g (h.descCoconeMorphism s)
    -/
    intro g; ext; exact h.uniq _ _ g.w
                  /-
                    🎉 no goals
                  -/
  this.trans this.symm


/-- Restating the definition of a colimit cocone in terms of the ∃! operator. -/
theorem existsUnique {t : Cocone F} (h : IsColimit t) (s : Cocone F) :
    ∃! d : t.pt ⟶ s.pt, ∀ j, t.ι.app j ≫ d = s.ι.app j :=
  ⟨h.desc s, h.fac s, h.uniq s⟩


/-- Noncomputably make a colimit cocone from the existence of unique factorizations. -/
def ofExistsUnique {t : Cocone F}
    (ht : ∀ s : Cocone F, ∃! d : t.pt ⟶ s.pt, ∀ j, t.ι.app j ≫ d = s.ι.app j) : IsColimit t := by
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cocone F
    ht : ∀ (s : CategoryTheory.Limits.Cocone F), ExistsUnique fun d => ∀ (j : J),  …
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  choose s hs hs' using ht
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cocone F
    s : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t.pt s.pt
    hs : ∀ (s_1 : CategoryTheory.Limits.Cocone F), (fun d => ∀ (j : J), Eq (Catego …
    hs' : ∀ (s_1 : CategoryTheory.Limits.Cocone F) (y : Quiver.Hom t.pt s_1.pt), ( …
    ⊢ CategoryTheory.Limits.IsColimit t
  -/
  exact ⟨s, hs, hs'⟩
  /-
    🎉 no goals
  -/


/-- Alternative constructor for `IsColimit`,
providing a morphism of cocones rather than a morphism between the cocone points
and separately the factorisation condition.
-/
@[simps]
def mkCoconeMorphism {t : Cocone F} (desc : ∀ s : Cocone F, t ⟶ s)
    (uniq' : ∀ (s : Cocone F) (m : t ⟶ s), m = desc s) : IsColimit t where
  desc s := (desc s).hom
  uniq s m w :=
                                                /-
                                                  J : Type u₁
                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                  K : Type u₂
                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                  C : Type u₃
                                                  inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                  F : CategoryTheory.Functor J C
                                                  t : CategoryTheory.Limits.Cocone F
                                                  desc : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t s
                                                  uniq' : ∀ (s : CategoryTheory.Limits.Cocone F) (m : Quiver.Hom t s), Eq m (des …
                                                  s : CategoryTheory.Limits.Cocone F
                                                  m : Quiver.Hom t.pt s.pt
                                                  w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m) (s.ι.app j)
                                                  ⊢ Eq { hom := m, w := w } (desc s)
                                                -/
    have : CoconeMorphism.mk m w = desc s := by apply uniq'
                                                /-
                                                  🎉 no goals
                                                -/
    congrArg CoconeMorphism.hom this


/-- Colimit cocones on `F` are unique up to isomorphism. -/
@[simps]
def uniqueUpToIso {s t : Cocone F} (P : IsColimit s) (Q : IsColimit t) : s ≅ t where
  hom := P.descCoconeMorphism t
  inv := Q.descCoconeMorphism s
  hom_inv_id := P.uniq_cocone_morphism
  inv_hom_id := Q.uniq_cocone_morphism


/-- Any cocone morphism between colimit cocones is an isomorphism. -/
theorem hom_isIso {s t : Cocone F} (P : IsColimit s) (Q : IsColimit t) (f : s ⟶ t) : IsIso f :=
  ⟨⟨Q.descCoconeMorphism s, ⟨P.uniq_cocone_morphism, Q.uniq_cocone_morphism⟩⟩⟩


/-- Colimits of `F` are unique up to isomorphism. -/
def coconePointUniqueUpToIso {s t : Cocone F} (P : IsColimit s) (Q : IsColimit t) : s.pt ≅ t.pt :=
  (Cocones.forget F).mapIso (uniqueUpToIso P Q)


@[reassoc (attr := simp)]
theorem comp_coconePointUniqueUpToIso_hom {s t : Cocone F} (P : IsColimit s) (Q : IsColimit t)
    (j : J) : s.ι.app j ≫ (coconePointUniqueUpToIso P Q).hom = t.ι.app j :=
  (uniqueUpToIso P Q).hom.w _


@[reassoc (attr := simp)]
theorem comp_coconePointUniqueUpToIso_inv {s t : Cocone F} (P : IsColimit s) (Q : IsColimit t)
    (j : J) : t.ι.app j ≫ (coconePointUniqueUpToIso P Q).inv = s.ι.app j :=
  (uniqueUpToIso P Q).inv.w _


@[reassoc (attr := simp)]
theorem coconePointUniqueUpToIso_hom_desc {r s t : Cocone F} (P : IsColimit s) (Q : IsColimit t) :
    (coconePointUniqueUpToIso P Q).hom ≫ Q.desc r = P.desc r :=
                 /-
                   J : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                   C : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   r s t : CategoryTheory.Limits.Cocone F
                   P : CategoryTheory.Limits.IsColimit s
                   Q : CategoryTheory.Limits.IsColimit t
                   ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheor …
                 -/
  P.uniq _ _ (by simp)
                 /-
                   🎉 no goals
                 -/


@[reassoc (attr := simp)]
theorem coconePointUniqueUpToIso_inv_desc {r s t : Cocone F} (P : IsColimit s) (Q : IsColimit t) :
    (coconePointUniqueUpToIso P Q).inv ≫ P.desc r = Q.desc r :=
                 /-
                   J : Type u₁
                   inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                   C : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   r s t : CategoryTheory.Limits.Cocone F
                   P : CategoryTheory.Limits.IsColimit s
                   Q : CategoryTheory.Limits.IsColimit t
                   ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (CategoryTheor …
                 -/
  Q.uniq _ _ (by simp)
                 /-
                   🎉 no goals
                 -/


/-- Transport evidence that a cocone is a colimit cocone across an isomorphism of cocones. -/
def ofIsoColimit {r t : Cocone F} (P : IsColimit r) (i : r ≅ t) : IsColimit t :=
  IsColimit.mkCoconeMorphism (fun s => i.inv ≫ P.descCoconeMorphism s) fun s m => by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cocone F
      P : CategoryTheory.Limits.IsColimit r
      i : CategoryTheory.Iso r t
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom t s
      ⊢ Eq m ((fun s => CategoryTheory.CategoryStruct.comp i.inv (P.descCoconeMorphi …
    -/
    rw [i.eq_inv_comp]; apply P.uniq_cocone_morphism
                        /-
                          🎉 no goals
                        -/


@[simp]
theorem ofIsoColimit_desc {r t : Cocone F} (P : IsColimit r) (i : r ≅ t) (s) :
    (P.ofIsoColimit i).desc s = i.inv.hom ≫ P.desc s :=
  rfl


/-- Isomorphism of cocones preserves whether or not they are colimiting cocones. -/
def equivIsoColimit {r t : Cocone F} (i : r ≅ t) : IsColimit r ≃ IsColimit t where
  toFun h := h.ofIsoColimit i
  invFun h := h.ofIsoColimit i.symm
                 /-
                   J : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} J
                   K : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                   C : Type u₃
                   inst✝ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   r t : CategoryTheory.Limits.Cocone F
                   i : CategoryTheory.Iso r t
                   ⊢ Function.LeftInverse (fun h => h.ofIsoColimit i.symm) fun h => h.ofIsoColimi …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    J : Type u₁
                    inst✝² : CategoryTheory.Category.{v₁, u₁} J
                    K : Type u₂
                    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                    C : Type u₃
                    inst✝ : CategoryTheory.Category.{v₃, u₃} C
                    F : CategoryTheory.Functor J C
                    r t : CategoryTheory.Limits.Cocone F
                    i : CategoryTheory.Iso r t
                    ⊢ Function.RightInverse (fun h => h.ofIsoColimit i.symm) fun h => h.ofIsoColim …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem equivIsoColimit_apply {r t : Cocone F} (i : r ≅ t) (P : IsColimit r) :
    equivIsoColimit i P = P.ofIsoColimit i :=
  rfl


@[simp]
theorem equivIsoColimit_symm_apply {r t : Cocone F} (i : r ≅ t) (P : IsColimit t) :
    (equivIsoColimit i).symm P = P.ofIsoColimit i.symm :=
  rfl


/-- If the canonical morphism to a cocone point from a colimiting cocone point is an iso, then the
first cocone was colimiting also.
-/
def ofPointIso {r t : Cocone F} (P : IsColimit r) [i : IsIso (P.desc t)] : IsColimit t :=
  ofIsoColimit P (by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cocone F
      P : CategoryTheory.Limits.IsColimit r
      i : CategoryTheory.IsIso (P.desc t)
      ⊢ CategoryTheory.Iso r t
    -/
    haveI : IsIso (P.descCoconeMorphism t).hom := i
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cocone F
      P : CategoryTheory.Limits.IsColimit r
      i : CategoryTheory.IsIso (P.desc t)
      this : CategoryTheory.IsIso (P.descCoconeMorphism t).hom
      ⊢ CategoryTheory.Iso r t
    -/
    haveI : IsIso (P.descCoconeMorphism t) := Cocones.cocone_iso_of_hom_iso _
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      r t : CategoryTheory.Limits.Cocone F
      P : CategoryTheory.Limits.IsColimit r
      i : CategoryTheory.IsIso (P.desc t)
      this✝ : CategoryTheory.IsIso (P.descCoconeMorphism t).hom
      this : CategoryTheory.IsIso (P.descCoconeMorphism t)
      ⊢ CategoryTheory.Iso r t
    -/
    apply asIso (P.descCoconeMorphism t))
    /-
      🎉 no goals
    -/


theorem hom_desc (h : IsColimit t) {W : C} (m : t.pt ⟶ W) :
    m =
      h.desc
        { pt := W
          ι :=
            { app := fun b => t.ι.app b ≫ m
                               /-
                                 J : Type u₁
                                 inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                 K : Type u₂
                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                 C : Type u₃
                                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                 F : CategoryTheory.Functor J C
                                 t : CategoryTheory.Limits.Cocone F
                                 h : CategoryTheory.Limits.IsColimit t
                                 W : C
                                 m : Quiver.Hom t.pt W
                                 ⊢ ∀ ⦃X Y : J⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp (F. …
                               -/
              naturality := by intros; erw [← assoc, t.ι.naturality, comp_id, comp_id] } } :=
                                       /-
                                         🎉 no goals
                                       -/
  h.uniq
    { pt := W
      ι :=
        { app := fun b => t.ι.app b ≫ m
          naturality := _ } }
    m fun _ => rfl


/-- Two morphisms out of a colimit are equal if their compositions with
  each cocone morphism are equal. -/
theorem hom_ext (h : IsColimit t) {W : C} {f f' : t.pt ⟶ W}
    (w : ∀ j, t.ι.app j ≫ f = t.ι.app j ≫ f') : f = f' := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    W : C
    f f' : Quiver.Hom t.pt W
    w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) f) (Category …
    ⊢ Eq f f'
  -/
  rw [h.hom_desc f, h.hom_desc f']; congr; exact funext w
                                           /-
                                             🎉 no goals
                                           -/


/-- Given a left adjoint functor between categories of cocones,
the image of a colimit cocone is a colimit cocone.
-/
def ofLeftAdjoint {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D} {left : Cocone G ⥤ Cocone F}
    {right : Cocone F ⥤ Cocone G} (adj : left ⊣ right) {c : Cocone G} (t : IsColimit c) :
    IsColimit (left.obj c) :=
  mkCoconeMorphism
    (fun s => (adj.homEquiv c s).symm (t.descCoconeMorphism _)) fun _ _ =>
    (Adjunction.homEquiv_apply_eq _ _ _).1 t.uniq_cocone_morphism


/-- Given two functors which have equivalent categories of cocones,
we can transport a colimiting cocone across the equivalence.
-/
def ofCoconeEquiv {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D} (h : Cocone G ≌ Cocone F)
    {c : Cocone G} : IsColimit (h.functor.obj c) ≃ IsColimit c where
  toFun P := ofIsoColimit (ofLeftAdjoint h.symm.toAdjunction P) (h.unitIso.symm.app c)
  invFun := ofLeftAdjoint h.toAdjunction
                 /-
                   J : Type u₁
                   inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                   K : Type u₂
                   inst✝² : CategoryTheory.Category.{v₂, u₂} K
                   C : Type u₃
                   inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                   F : CategoryTheory.Functor J C
                   t : CategoryTheory.Limits.Cocone F
                   D : Type u₄
                   inst✝ : CategoryTheory.Category.{v₄, u₄} D
                   G : CategoryTheory.Functor K D
                   h : CategoryTheory.Equivalence (CategoryTheory.Limits.Cocone G) (CategoryTheor …
                   c : CategoryTheory.Limits.Cocone G
                   ⊢ Function.LeftInverse (CategoryTheory.Limits.IsColimit.ofLeftAdjoint h.toAdju …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    J : Type u₁
                    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
                    K : Type u₂
                    inst✝² : CategoryTheory.Category.{v₂, u₂} K
                    C : Type u₃
                    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
                    F : CategoryTheory.Functor J C
                    t : CategoryTheory.Limits.Cocone F
                    D : Type u₄
                    inst✝ : CategoryTheory.Category.{v₄, u₄} D
                    G : CategoryTheory.Functor K D
                    h : CategoryTheory.Equivalence (CategoryTheory.Limits.Cocone G) (CategoryTheor …
                    c : CategoryTheory.Limits.Cocone G
                    ⊢ Function.RightInverse (CategoryTheory.Limits.IsColimit.ofLeftAdjoint h.toAdj …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem ofCoconeEquiv_apply_desc {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D}
    (h : Cocone G ≌ Cocone F) {c : Cocone G} (P : IsColimit (h.functor.obj c)) (s) :
    (ofCoconeEquiv h P).desc s =
      (h.unit.app c).hom ≫
        (h.inverse.map (P.descCoconeMorphism (h.functor.obj s))).hom ≫ (h.unitInv.app s).hom :=
  rfl


@[simp]
theorem ofCoconeEquiv_symm_apply_desc {D : Type u₄} [Category.{v₄} D] {G : K ⥤ D}
    (h : Cocone G ≌ Cocone F) {c : Cocone G} (P : IsColimit c) (s) :
    ((ofCoconeEquiv h).symm P).desc s =
      (h.functor.map (P.descCoconeMorphism (h.inverse.obj s))).hom ≫ (h.counit.app s).hom :=
  rfl


/-- A cocone precomposed with a natural isomorphism is a colimit cocone
if and only if the original cocone is.
-/
def precomposeHomEquiv {F G : J ⥤ C} (α : F ≅ G) (c : Cocone G) :
    IsColimit ((Cocones.precompose α.hom).obj c) ≃ IsColimit c :=
  ofCoconeEquiv (Cocones.precomposeEquivalence α)


/-- A cocone precomposed with the inverse of a natural isomorphism is a colimit cocone
if and only if the original cocone is.
-/
def precomposeInvEquiv {F G : J ⥤ C} (α : F ≅ G) (c : Cocone F) :
    IsColimit ((Cocones.precompose α.inv).obj c) ≃ IsColimit c :=
  precomposeHomEquiv α.symm c


/-- Constructing an equivalence `is_colimit c ≃ is_colimit d` from a natural isomorphism
between the underlying functors, and then an isomorphism between `c` transported along this and `d`.
-/
def equivOfNatIsoOfIso {F G : J ⥤ C} (α : F ≅ G) (c : Cocone F) (d : Cocone G)
    (w : (Cocones.precompose α.inv).obj c ≅ d) : IsColimit c ≃ IsColimit d :=
  (precomposeInvEquiv α _).symm.trans (equivIsoColimit w)


/-- The cocone points of two colimit cocones for naturally isomorphic functors
are themselves isomorphic.
-/
@[simps]
def coconePointsIsoOfNatIso {F G : J ⥤ C} {s : Cocone F} {t : Cocone G} (P : IsColimit s)
    (Q : IsColimit t) (w : F ≅ G) : s.pt ≅ t.pt where
  hom := P.map t w.hom
  inv := Q.map s w.inv
                              /-
                                J : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                K : Type u₂
                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                C : Type u₃
                                inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                F✝ : CategoryTheory.Functor J C
                                t✝ : CategoryTheory.Limits.Cocone F✝
                                F G : CategoryTheory.Functor J C
                                s : CategoryTheory.Limits.Cocone F
                                t : CategoryTheory.Limits.Cocone G
                                P : CategoryTheory.Limits.IsColimit s
                                Q : CategoryTheory.Limits.IsColimit t
                                w : CategoryTheory.Iso F G
                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheor …
                              -/
  hom_inv_id := P.hom_ext (by aesop_cat)
                              /-
                                🎉 no goals
                              -/
                              /-
                                J : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                K : Type u₂
                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                C : Type u₃
                                inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                F✝ : CategoryTheory.Functor J C
                                t✝ : CategoryTheory.Limits.Cocone F✝
                                F G : CategoryTheory.Functor J C
                                s : CategoryTheory.Limits.Cocone F
                                t : CategoryTheory.Limits.Cocone G
                                P : CategoryTheory.Limits.IsColimit s
                                Q : CategoryTheory.Limits.IsColimit t
                                w : CategoryTheory.Iso F G
                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (CategoryTheor …
                              -/
  inv_hom_id := Q.hom_ext (by aesop_cat)
                              /-
                                🎉 no goals
                              -/


@[reassoc]
theorem comp_coconePointsIsoOfNatIso_hom {F G : J ⥤ C} {s : Cocone F} {t : Cocone G}
    (P : IsColimit s) (Q : IsColimit t) (w : F ≅ G) (j : J) :
                                                                                    /-
                                                                                      J : Type u₁
                                                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                                      C : Type u₃
                                                                                      inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                                      F G : CategoryTheory.Functor J C
                                                                                      s : CategoryTheory.Limits.Cocone F
                                                                                      t : CategoryTheory.Limits.Cocone G
                                                                                      P : CategoryTheory.Limits.IsColimit s
                                                                                      Q : CategoryTheory.Limits.IsColimit t
                                                                                      w : CategoryTheory.Iso F G
                                                                                      j : J
                                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (P.coconePointsIsoOfNatIs …
                                                                                    -/
    s.ι.app j ≫ (coconePointsIsoOfNatIso P Q w).hom = w.hom.app j ≫ t.ι.app j := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[reassoc]
theorem comp_coconePointsIsoOfNatIso_inv {F G : J ⥤ C} {s : Cocone F} {t : Cocone G}
    (P : IsColimit s) (Q : IsColimit t) (w : F ≅ G) (j : J) :
                                                                                    /-
                                                                                      J : Type u₁
                                                                                      inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                                                                                      C : Type u₃
                                                                                      inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                                      F G : CategoryTheory.Functor J C
                                                                                      s : CategoryTheory.Limits.Cocone F
                                                                                      t : CategoryTheory.Limits.Cocone G
                                                                                      P : CategoryTheory.Limits.IsColimit s
                                                                                      Q : CategoryTheory.Limits.IsColimit t
                                                                                      w : CategoryTheory.Iso F G
                                                                                      j : J
                                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (P.coconePointsIsoOfNatIs …
                                                                                    -/
    t.ι.app j ≫ (coconePointsIsoOfNatIso P Q w).inv = w.inv.app j ≫ s.ι.app j := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[reassoc]
theorem coconePointsIsoOfNatIso_hom_desc {F G : J ⥤ C} {s : Cocone F} {r t : Cocone G}
    (P : IsColimit s) (Q : IsColimit t) (w : F ≅ G) :
    (coconePointsIsoOfNatIso P Q w).hom ≫ Q.desc r = P.map _ w.hom :=
                /-
                  J : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                  C : Type u₃
                  inst✝ : CategoryTheory.Category.{v₃, u₃} C
                  F G : CategoryTheory.Functor J C
                  s : CategoryTheory.Limits.Cocone F
                  r t : CategoryTheory.Limits.Cocone G
                  P : CategoryTheory.Limits.IsColimit s
                  Q : CategoryTheory.Limits.IsColimit t
                  w : CategoryTheory.Iso F G
                  ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheor …
                -/
  P.hom_ext (by simp)
                /-
                  🎉 no goals
                -/


@[reassoc]
theorem coconePointsIsoOfNatIso_inv_desc {F G : J ⥤ C} {s : Cocone G} {r t : Cocone F}
    (P : IsColimit t) (Q : IsColimit s) (w : F ≅ G) :
    (coconePointsIsoOfNatIso P Q w).inv ≫ P.desc r = Q.map _ w.inv :=
                /-
                  J : Type u₁
                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
                  C : Type u₃
                  inst✝ : CategoryTheory.Category.{v₃, u₃} C
                  F G : CategoryTheory.Functor J C
                  s : CategoryTheory.Limits.Cocone G
                  r t : CategoryTheory.Limits.Cocone F
                  P : CategoryTheory.Limits.IsColimit t
                  Q : CategoryTheory.Limits.IsColimit s
                  w : CategoryTheory.Iso F G
                  ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheor …
                -/
  Q.hom_ext (by simp)
                /-
                  🎉 no goals
                -/


/-- If `s : Cocone F` is a colimit cocone, so is `s` whiskered by an equivalence `e`.
-/
def whiskerEquivalence {s : Cocone F} (P : IsColimit s) (e : K ≌ J) :
    IsColimit (s.whisker e.functor) :=
  ofLeftAdjoint (Cocones.whiskeringEquivalence e).toAdjunction P


/-- If `s : Cocone F` whiskered by an equivalence `e` is a colimit cocone, so is `s`.
-/
def ofWhiskerEquivalence {s : Cocone F} (e : K ≌ J) (P : IsColimit (s.whisker e.functor)) :
    IsColimit s :=
  equivIsoColimit ((Cocones.whiskeringEquivalence e).unitIso.app s).symm
    (ofLeftAdjoint (Cocones.whiskeringEquivalence e).symm.toAdjunction P)


/-- Given an equivalence of diagrams `e`, `s` is a colimit cocone iff `s.whisker e.functor` is.
-/
def whiskerEquivalenceEquiv {s : Cocone F} (e : K ≌ J) :
    IsColimit s ≃ IsColimit (s.whisker e.functor) :=
                                                               /-
                                                                 J : Type u₁
                                                                 inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                                 K : Type u₂
                                                                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                                 C : Type u₃
                                                                 inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                                 F : CategoryTheory.Functor J C
                                                                 t s : CategoryTheory.Limits.Cocone F
                                                                 e : CategoryTheory.Equivalence K J
                                                                 ⊢ Function.LeftInverse (CategoryTheory.Limits.IsColimit.ofWhiskerEquivalence e …
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  ⟨fun h => h.whiskerEquivalence e, ofWhiskerEquivalence e, by aesop_cat, by aesop_cat⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- A colimit cocone extended by an isomorphism is a colimit cocone. -/
def extendIso {s : Cocone F} {X : C} (i : s.pt ⟶ X) [IsIso i] (hs : IsColimit s) :
    IsColimit (s.extend i) :=
  IsColimit.ofIsoColimit hs (Cocones.extendIso s (asIso i))


/-- A cocone is a colimit cocone if its extension by an isomorphism is. -/
def ofExtendIso {s : Cocone F} {X : C} (i : s.pt ⟶ X) [IsIso i] (hs : IsColimit (s.extend i)) :
    IsColimit s :=
  IsColimit.ofIsoColimit hs (Cocones.extendIso s (asIso i)).symm


/-- A cocone is a colimit cocone iff its extension by an isomorphism is. -/
def extendIsoEquiv {s : Cocone F} {X : C} (i : s.pt ⟶ X) [IsIso i] :
    IsColimit s ≃ IsColimit (s.extend i) :=
  equivOfSubsingletonOfSubsingleton (extendIso i) (ofExtendIso i)


/-- We can prove two cocone points `(s : Cocone F).pt` and `(t : Cocone G).pt` are isomorphic if
* both cocones are colimit cocones
* their indexing categories are equivalent via some `e : J ≌ K`,
* the triangle of functors commutes up to a natural isomorphism: `e.functor ⋙ G ≅ F`.

This is the most general form of uniqueness of cocone points,
allowing relabelling of both the indexing category (up to equivalence)
and the functor (up to natural isomorphism).
-/
@[simps]
def coconePointsIsoOfEquivalence {F : J ⥤ C} {s : Cocone F} {G : K ⥤ C} {t : Cocone G}
    (P : IsColimit s) (Q : IsColimit t) (e : J ≌ K) (w : e.functor ⋙ G ≅ F) : s.pt ≅ t.pt :=
  let w' : e.inverse ⋙ F ≅ G := (isoWhiskerLeft e.inverse w).symm ≪≫ invFunIdAssoc e G
  { hom := P.desc ((Cocones.equivalenceOfReindexing e w).functor.obj t)
    inv := Q.desc ((Cocones.equivalenceOfReindexing e.symm w').functor.obj s)
    hom_inv_id := by
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (P.desc ((CategoryTheory.Limits.Cocon …
      -/
      apply hom_ext P; intro j
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheory.CategoryS …
      -/
      dsimp [w']
      simp only [Limits.Cocone.whisker_ι, fac, invFunIdAssoc_inv_app, whiskerLeft_app, assoc,
        comp_id, Limits.Cocones.precompose_obj_ι, fac_assoc, NatTrans.comp_app]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (w.inv.app j) (CategoryTheory.Categor …
      -/
      rw [counitInv_app_functor, ← Functor.comp_map, ← w.inv.naturality_assoc]
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (e.unit.app j)) (CategoryTheor …
      -/
      dsimp
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (e.unit.app j)) (CategoryTheor …
      -/
      simp
      /-
        🎉 no goals
      -/
    inv_hom_id := by
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Q.desc ((CategoryTheory.Limits.Cocon …
      -/
      apply hom_ext Q
      /-
        J : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} C
        F✝ : CategoryTheory.Functor J C
        t✝ : CategoryTheory.Limits.Cocone F✝
        F : CategoryTheory.Functor J C
        s : CategoryTheory.Limits.Cocone F
        G : CategoryTheory.Functor K C
        t : CategoryTheory.Limits.Cocone G
        P : CategoryTheory.Limits.IsColimit s
        Q : CategoryTheory.Limits.IsColimit t
        e : CategoryTheory.Equivalence J K
        w : CategoryTheory.Iso (e.functor.comp G) F
        w' : CategoryTheory.Iso (e.inverse.comp F) G := (CategoryTheory.isoWhiskerLeft …
        ⊢ ∀ (j : K), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (CategoryTheor …
      -/
      aesop_cat }
      /-
        🎉 no goals
      -/


/-- The universal property of a colimit cocone: a map `X ⟶ W` is the same as
  a cocone on `F` with cone point `W`. -/
def homEquiv (h : IsColimit t) (W : C) : (t.pt ⟶ W) ≃ (F ⟶ (const J).obj W) where
  toFun f := (t.extend f).ι
  invFun ι := h.desc
      { pt := W
        ι }
                              /-
                                J : Type u₁
                                inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                K : Type u₂
                                inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                C : Type u₃
                                inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                F : CategoryTheory.Functor J C
                                t : CategoryTheory.Limits.Cocone F
                                h : CategoryTheory.Limits.IsColimit t
                                W : C
                                f : Quiver.Hom t.pt W
                                ⊢ ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) ((fun ι => h.d …
                              -/
  left_inv f := h.hom_ext (by simp)
                              /-
                                🎉 no goals
                              -/
                    /-
                      J : Type u₁
                      inst✝² : CategoryTheory.Category.{v₁, u₁} J
                      K : Type u₂
                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                      C : Type u₃
                      inst✝ : CategoryTheory.Category.{v₃, u₃} C
                      F : CategoryTheory.Functor J C
                      t : CategoryTheory.Limits.Cocone F
                      h : CategoryTheory.Limits.IsColimit t
                      W : C
                      ι : Quiver.Hom F ((CategoryTheory.Functor.const J).obj W)
                      ⊢ Eq ((fun f => (t.extend f).ι) ((fun ι => h.desc { pt := W, ι := ι }) ι)) ι
                    -/
  right_inv ι := by aesop_cat
                    /-
                      🎉 no goals
                    -/


@[simp]
lemma homEquiv_apply (h : IsColimit t) {W : C} (f : t.pt ⟶ W) :
    h.homEquiv W f = (t.extend f).ι := rfl


/-- The universal property of a colimit cocone: a map `X ⟶ W` is the same as
  a cocone on `F` with cone point `W`. -/
def homIso (h : IsColimit t) (W : C) : ULift.{u₁} (t.pt ⟶ W : Type v₃) ≅ F ⟶ (const J).obj W :=
  Equiv.toIso (Equiv.ulift.trans (h.homEquiv W))


@[simp]
theorem homIso_hom (h : IsColimit t) {W : C} (f : ULift (t.pt ⟶ W)) :
    (IsColimit.homIso h W).hom f = (t.extend f.down).ι :=
  rfl


/-- The colimit of `F` represents the functor taking `W` to
  the set of cocones on `F` with cone point `W`. -/
def natIso (h : IsColimit t) : coyoneda.obj (op t.pt) ⋙ uliftFunctor.{u₁} ≅ F.cocones :=
  /-
    J : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} J
    K : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    t : CategoryTheory.Limits.Cocone F
    h : CategoryTheory.Limits.IsColimit t
    ⊢ ∀ {X Y : C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((( …
  -/
  NatIso.ofComponents (IsColimit.homIso h)
  /-
    🎉 no goals
  -/


/-- Another, more explicit, formulation of the universal property of a colimit cocone.
See also `homIso`. -/
def homIso' (h : IsColimit t) (W : C) :
    ULift.{u₁} (t.pt ⟶ W : Type v₃) ≅
      { p : ∀ j, F.obj j ⟶ W // ∀ {j j' : J} (f : j ⟶ j'), F.map f ≫ p j' = p j } :=
  h.homIso W ≪≫
    { hom := fun ι =>
                                                /-
                                                  J : Type u₁
                                                  inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                                  K : Type u₂
                                                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                                  C : Type u₃
                                                  inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                                  F : CategoryTheory.Functor J C
                                                  t : CategoryTheory.Limits.Cocone F
                                                  h : CategoryTheory.Limits.IsColimit t
                                                  W : C
                                                  ι : Quiver.Hom F ((CategoryTheory.Functor.const J).obj W)
                                                  j j' : J
                                                  f : Quiver.Hom j j'
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => ι.app j) j')) (( …
                                                -/
        ⟨fun j => ι.app j, fun {j} {j'} f => by convert ← ι.naturality f; apply comp_id⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
      inv := fun p =>
        { app := fun j => p.1 j
                                         /-
                                           J : Type u₁
                                           inst✝² : CategoryTheory.Category.{v₁, u₁} J
                                           K : Type u₂
                                           inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
                                           C : Type u₃
                                           inst✝ : CategoryTheory.Category.{v₃, u₃} C
                                           F : CategoryTheory.Functor J C
                                           t : CategoryTheory.Limits.Cocone F
                                           h : CategoryTheory.Limits.IsColimit t
                                           W : C
                                           p : Subtype fun p => ∀ {j j' : J} (f : Quiver.Hom j j'), Eq (CategoryTheory.Ca …
                                           j j' : J
                                           f : Quiver.Hom j j'
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) ((fun j => ↑p j) j')) (Cate …
                                         -/
          naturality := fun j j' f => by dsimp; rw [comp_id]; exact p.2 f } }
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If G : C → D is a faithful functor which sends t to a colimit cocone,
  then it suffices to check that the induced maps for the image of t
  can be lifted to maps of C. -/
def ofFaithful {t : Cocone F} {D : Type u₄} [Category.{v₄} D] (G : C ⥤ D) [G.Faithful]
    (ht : IsColimit (mapCocone G t)) (desc : ∀ s : Cocone F, t.pt ⟶ s.pt)
    (h : ∀ s, G.map (desc s) = ht.desc (mapCocone G s)) : IsColimit t :=
  { desc
                         /-
                           J : Type u₁
                           inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
                           K : Type u₂
                           inst✝³ : CategoryTheory.Category.{v₂, u₂} K
                           C : Type u₃
                           inst✝² : CategoryTheory.Category.{v₃, u₃} C
                           F : CategoryTheory.Functor J C
                           t✝ t : CategoryTheory.Limits.Cocone F
                           D : Type u₄
                           inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
                           G : CategoryTheory.Functor C D
                           inst✝ : G.Faithful
                           ht : CategoryTheory.Limits.IsColimit (G.mapCocone t)
                           desc : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t.pt s.pt
                           h : ∀ (s : CategoryTheory.Limits.Cocone F), Eq (G.map (desc s)) (ht.desc (G.ma …
                           s : CategoryTheory.Limits.Cocone F
                           j : J
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (desc s)) (s.ι.app j)
                         -/
    fac := fun s j => by apply G.map_injective; rw [G.map_comp, h]; apply ht.fac
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    uniq := fun s m w => by
      /-
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cocone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsColimit (G.mapCocone t)
        desc : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t.pt s.pt
        h : ∀ (s : CategoryTheory.Limits.Cocone F), Eq (G.map (desc s)) (ht.desc (G.ma …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m) (s.ι.app j)
        ⊢ Eq m (desc s)
      -/
      apply G.map_injective; rw [h]
      /-
        case a
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cocone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsColimit (G.mapCocone t)
        desc : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t.pt s.pt
        h : ∀ (s : CategoryTheory.Limits.Cocone F), Eq (G.map (desc s)) (ht.desc (G.ma …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m) (s.ι.app j)
        ⊢ Eq (G.map m) (ht.desc (G.mapCocone s))
      -/
      refine ht.uniq (mapCocone G s) _ fun j => ?_
      /-
        case a
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cocone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsColimit (G.mapCocone t)
        desc : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t.pt s.pt
        h : ∀ (s : CategoryTheory.Limits.Cocone F), Eq (G.map (desc s)) (ht.desc (G.ma …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m) (s.ι.app j)
        j : J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.mapCocone t).ι.app j) (G.map m))  …
      -/
      convert ← congrArg (fun f => G.map f) (w j)
      /-
        case h.e'_2.h
        J : Type u₁
        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} J
        K : Type u₂
        inst✝³ : CategoryTheory.Category.{v₂, u₂} K
        C : Type u₃
        inst✝² : CategoryTheory.Category.{v₃, u₃} C
        F : CategoryTheory.Functor J C
        t✝ t : CategoryTheory.Limits.Cocone F
        D : Type u₄
        inst✝¹ : CategoryTheory.Category.{v₄, u₄} D
        G : CategoryTheory.Functor C D
        inst✝ : G.Faithful
        ht : CategoryTheory.Limits.IsColimit (G.mapCocone t)
        desc : (s : CategoryTheory.Limits.Cocone F) → Quiver.Hom t.pt s.pt
        h : ∀ (s : CategoryTheory.Limits.Cocone F), Eq (G.map (desc s)) (ht.desc (G.ma …
        s : CategoryTheory.Limits.Cocone F
        m : Quiver.Hom t.pt s.pt
        w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) m) (s.ι.app j)
        j : J
        e_1✝ : Eq (Quiver.Hom (G.obj (F.obj j)) (G.obj s.pt)) (Quiver.Hom ((F.comp G). …
        ⊢ Eq (G.map (CategoryTheory.CategoryStruct.comp (t.ι.app j) m)) (CategoryTheor …
      -/
      apply G.map_comp }
      /-
        🎉 no goals
      -/


/-- If `F` and `G` are naturally isomorphic, then `F.mapCocone c` being a colimit implies
`G.mapCocone c` is also a colimit.
-/
def mapCoconeEquiv {D : Type u₄} [Category.{v₄} D] {K : J ⥤ C} {F G : C ⥤ D} (h : F ≅ G)
    {c : Cocone K} (t : IsColimit (mapCocone F c)) : IsColimit (mapCocone G c) := by
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    F✝ : CategoryTheory.Functor J C
    t✝ : CategoryTheory.Limits.Cocone F✝
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    K : CategoryTheory.Functor J C
    F G : CategoryTheory.Functor C D
    h : CategoryTheory.Iso F G
    c : CategoryTheory.Limits.Cocone K
    t : CategoryTheory.Limits.IsColimit (F.mapCocone c)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone c)
  -/
  apply IsColimit.ofIsoColimit _ (precomposeWhiskerLeftMapCocone h c)
  /-
    J : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} J
    K✝ : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} K✝
    C : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} C
    F✝ : CategoryTheory.Functor J C
    t✝ : CategoryTheory.Limits.Cocone F✝
    D : Type u₄
    inst✝ : CategoryTheory.Category.{v₄, u₄} D
    K : CategoryTheory.Functor J C
    F G : CategoryTheory.Functor C D
    h : CategoryTheory.Iso F G
    c : CategoryTheory.Limits.Cocone K
    t : CategoryTheory.Limits.IsColimit (F.mapCocone c)
    ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompose ( …
  -/
  apply (precomposeInvEquiv (isoWhiskerLeft K h : _) _).symm t
  /-
    🎉 no goals
  -/


/-- A cocone is a colimit cocone exactly if
there is a unique cocone morphism from any other cocone.
-/
def isoUniqueCoconeMorphism {t : Cocone F} : IsColimit t ≅ ∀ s, Unique (t ⟶ s) where
  hom h s :=
    { default := h.descCoconeMorphism s
      uniq := fun _ => h.uniq_cocone_morphism }
  inv h :=
    { desc := fun s => (h s).default.hom
      uniq := fun s f w => congrArg CoconeMorphism.hom ((h s).uniq ⟨f, w⟩) }


/-- If `F.cocones` is corepresented by `X`, each morphism `f : X ⟶ Y` gives a cocone with cone
point `Y`. -/
def coconeOfHom {Y : C} (f : X ⟶ Y) : Cocone F where
  pt := Y
  ι := h.hom.app Y ⟨f⟩


/-- If `F.cocones` is corepresented by `X`, each cocone `s` gives a morphism `X ⟶ s.pt`. -/
def homOfCocone (s : Cocone F) : X ⟶ s.pt :=
  (h.inv.app s.pt s.ι).down


@[simp]
theorem coconeOfHom_homOfCocone (s : Cocone F) : coconeOfHom h (homOfCocone h s) = s := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    ⊢ Eq (CategoryTheory.Limits.IsColimit.OfNatIso.coconeOfHom h (CategoryTheory.L …
  -/
  dsimp [coconeOfHom, homOfCocone]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    ⊢ Eq { pt := s.pt, ι := h.hom.app s.pt { down := (h.inv.app s.pt s.ι).down } } s
  -/
  have ⟨s_pt,s_ι⟩ := s
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    s_pt : C
    s_ι : Quiver.Hom F ((CategoryTheory.Functor.const J).obj s_pt)
    ⊢ Eq { pt := { pt := s_pt, ι := s_ι }.pt, ι := h.hom.app { pt := s_pt, ι := s_ …
  -/
  congr; dsimp
  /-
    case e_ι
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    s_pt : C
    s_ι : Quiver.Hom F ((CategoryTheory.Functor.const J).obj s_pt)
    ⊢ Eq (h.hom.app s_pt { down := (h.inv.app s_pt s_ι).down }) s_ι
  -/
  convert congrFun (congrFun (congrArg NatTrans.app h.inv_hom_id) s_pt) s_ι using 1
  /-
    🎉 no goals
  -/


@[simp]
theorem homOfCocone_cooneOfHom {Y : C} (f : X ⟶ Y) : homOfCocone h (coconeOfHom h f) = f :=
  congrArg ULift.down (congrFun (congrFun (congrArg NatTrans.app h.hom_inv_id) Y) ⟨f⟩ : _)


/-- If `F.cocones` is corepresented by `X`, the cocone corresponding to the identity morphism on `X`
will be a colimit cocone. -/
def colimitCocone : Cocone F :=
  coconeOfHom h (𝟙 X)


/-- If `F.cocones` is corepresented by `X`, the cocone corresponding to a morphism `f : Y ⟶ X` is
the colimit cocone extended by `f`. -/
theorem coconeOfHom_fac {Y : C} (f : X ⟶ Y) : coconeOfHom h f = (colimitCocone h).extend f := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Limits.IsColimit.OfNatIso.coconeOfHom h f) ((CategoryTheo …
  -/
  dsimp [coconeOfHom, colimitCocone, Cocone.extend]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    ⊢ Eq { pt := Y, ι := h.hom.app Y { down := f } } { pt := Y, ι := CategoryTheor …
  -/
  congr with j
  /-
    case e_ι.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    j : J
    ⊢ Eq ((h.hom.app Y { down := f }).app j) ((CategoryTheory.CategoryStruct.comp  …
  -/
  have t := congrFun (h.hom.naturality f) ⟨𝟙 X⟩
  /-
    case e_ι.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    j : J
    t : Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.coyoneda.obj { un …
    ⊢ Eq ((h.hom.app Y { down := f }).app j) ((CategoryTheory.CategoryStruct.comp  …
  -/
  dsimp at t
  /-
    case e_ι.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    j : J
    t : Eq (h.hom.app Y { down := CategoryTheory.CategoryStruct.comp (CategoryTheo …
    ⊢ Eq ((h.hom.app Y { down := f }).app j) ((CategoryTheory.CategoryStruct.comp  …
  -/
  simp only [id_comp] at t
  /-
    case e_ι.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    j : J
    t : Eq (h.hom.app Y { down := f }) (F.cocones.map f (h.hom.app X { down := Cat …
    ⊢ Eq ((h.hom.app Y { down := f }).app j) ((CategoryTheory.CategoryStruct.comp  …
  -/
  rw [congrFun (congrArg NatTrans.app t) j]
  /-
    case e_ι.w.h
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    Y : C
    f : Quiver.Hom X Y
    j : J
    t : Eq (h.hom.app Y { down := f }) (F.cocones.map f (h.hom.app X { down := Cat …
    ⊢ Eq ((F.cocones.map f (h.hom.app X { down := CategoryTheory.CategoryStruct.id …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `F.cocones` is corepresented by `X`, any cocone is the extension of the colimit cocone by the
corresponding morphism. -/
theorem cocone_fac (s : Cocone F) : (colimitCocone h).extend (homOfCocone h s) = s := by
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    ⊢ Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extend (Categ …
  -/
  rw [← coconeOfHom_homOfCocone h s]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    ⊢ Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extend (Categ …
  -/
  conv_lhs => simp only [homOfCocone_cooneOfHom]
  /-
    J : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} J
    C : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} C
    F : CategoryTheory.Functor J C
    X : C
    h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
    s : CategoryTheory.Limits.Cocone F
    ⊢ Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extend (Categ …
  -/
  apply (coconeOfHom_fac _ _).symm
  /-
    🎉 no goals
  -/


/-- If `F.cocones` is corepresentable, then the cocone corresponding to the identity morphism on
the representing object is a colimit cocone.
-/
def ofNatIso {X : C} (h : coyoneda.obj (op X) ⋙ uliftFunctor.{u₁} ≅ F.cocones) :
    IsColimit (colimitCocone h) where
  desc s := homOfCocone h s
  fac s j := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      s : CategoryTheory.Limits.Cocone F
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.IsColimit.OfN …
    -/
    have h := cocone_fac h s
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h✝ : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Cate …
      s : CategoryTheory.Limits.Cocone F
      j : J
      h : Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h✝).extend (Ca …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.IsColimit.OfN …
    -/
    cases s
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h✝ : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Cate …
      j : J
      pt✝ : C
      ι✝ : Quiver.Hom F ((CategoryTheory.Functor.const J).obj pt✝)
      h : Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h✝).extend (Ca …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.IsColimit.OfN …
    -/
    injection h with h₁ h₂
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      j : J
      pt✝ : C
      ι✝ : Quiver.Hom F ((CategoryTheory.Functor.const J).obj pt✝)
      h₁ : Eq { pt := pt✝, ι := ι✝ }.pt pt✝
      h₂ : Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extensions …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.IsColimit.OfN …
    -/
    simp only [heq_iff_eq] at h₂
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      j : J
      pt✝ : C
      ι✝ : Quiver.Hom F ((CategoryTheory.Functor.const J).obj pt✝)
      h₁ : Eq { pt := pt✝, ι := ι✝ }.pt pt✝
      h₂ : Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extensions …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.IsColimit.OfN …
    -/
    conv_rhs => rw [← h₂]
    /-
      case mk
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      j : J
      pt✝ : C
      ι✝ : Quiver.Hom F ((CategoryTheory.Functor.const J).obj pt✝)
      h₁ : Eq { pt := pt✝, ι := ι✝ }.pt pt✝
      h₂ : Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extensions …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.IsColimit.OfN …
    -/
    rfl
    /-
      🎉 no goals
    -/
  uniq s m w := by
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).pt s …
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq m ((fun s => CategoryTheory.Limits.IsColimit.OfNatIso.homOfCocone h s) s)
    -/
    rw [← homOfCocone_cooneOfHom h m]
    /-
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).pt s …
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq (CategoryTheory.Limits.IsColimit.OfNatIso.homOfCocone h (CategoryTheory.L …
    -/
    congr
    /-
      case h.e_8.h
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).pt s …
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq (CategoryTheory.Limits.IsColimit.OfNatIso.coconeOfHom h m) s
    -/
    rw [coconeOfHom_fac]
    /-
      case h.e_8.h
      J : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} J
      K : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} K
      C : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} C
      F : CategoryTheory.Functor J C
      t : CategoryTheory.Limits.Cocone F
      X : C
      h : CategoryTheory.Iso ((CategoryTheory.coyoneda.obj { unop := X }).comp Categ …
      s : CategoryTheory.Limits.Cocone F
      m : Quiver.Hom (CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).pt s …
      w : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits. …
      ⊢ Eq ((CategoryTheory.Limits.IsColimit.OfNatIso.colimitCocone h).extend m) s
    -/
    dsimp [Cocone.extend]; cases s; congr with j; exact w j
                                                  /-
                                                    🎉 no goals
                                                  -/


