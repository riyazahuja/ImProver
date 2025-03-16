/-- Auxiliary definition for `HasCokernels SemiNormedGrp₁`. -/
def cokernelCocone {X Y : SemiNormedGrp₁.{u}} (f : X ⟶ Y) : Cofork f 0 :=
  Cofork.ofπ
    (@SemiNormedGrp₁.mkHom _ (SemiNormedGrp.of (Y ⧸ NormedAddGroupHom.range f.1))
      f.1.range.normedMk (NormedAddGroupHom.isQuotientQuotient _).norm_le)
    (by
      /-
        X Y : SemiNormedGrp₁
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (SemiNormedGrp₁.mkHom (↑f).range.no …
      -/
      ext x
      -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5026): was
      -- simp only [comp_apply, Limits.zero_comp, NormedAddGroupHom.zero_apply,
      --   SemiNormedGrp₁.mkHom_apply, SemiNormedGrp₁.zero_apply,
      --   ← NormedAddGroupHom.mem_ker, f.1.range.ker_normedMk, f.1.mem_range]
      -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
      erw [Limits.zero_comp, comp_apply, SemiNormedGrp₁.mkHom_apply,
        SemiNormedGrp₁.zero_apply, ← NormedAddGroupHom.mem_ker, f.1.range.ker_normedMk,
        f.1.mem_range]
      /-
        case w.h
        X Y : SemiNormedGrp₁
        f : Quiver.Hom X Y
        x : ↑X
        ⊢ Exists fun w => Eq (↑f w) (f x)
      -/
      use x
      /-
        case h
        X Y : SemiNormedGrp₁
        f : Quiver.Hom X Y
        x : ↑X
        ⊢ Eq (↑f x) (f x)
      -/
      rfl)
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `HasCokernels SemiNormedGrp₁`. -/
def cokernelLift {X Y : SemiNormedGrp₁.{u}} (f : X ⟶ Y) (s : CokernelCofork f) :
    (cokernelCocone f).pt ⟶ s.pt := by
  /-
    X Y : SemiNormedGrp₁
    f : Quiver.Hom X Y
    s : CategoryTheory.Limits.CokernelCofork f
    ⊢ Quiver.Hom (SemiNormedGrp₁.cokernelCocone f).pt s.pt
  -/
  fconstructor
  -- The lift itself:
    /-
      case val
      X Y : SemiNormedGrp₁
      f : Quiver.Hom X Y
      s : CategoryTheory.Limits.CokernelCofork f
      ⊢ NormedAddGroupHom ↑(SemiNormedGrp₁.cokernelCocone f).pt ↑s.pt
    -/
  · apply NormedAddGroupHom.lift _ s.π.1
    /-
      case val
      X Y : SemiNormedGrp₁
      f : Quiver.Hom X Y
      s : CategoryTheory.Limits.CokernelCofork f
      ⊢ ∀ (s_1 : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limit …
    -/
    rintro _ ⟨b, rfl⟩
    /-
      case val.intro
      X Y : SemiNormedGrp₁
      f : Quiver.Hom X Y
      s : CategoryTheory.Limits.CokernelCofork f
      b : ↑X
      ⊢ Eq (↑(CategoryTheory.Limits.Cofork.π s) ((↑f).toAddMonoidHom b)) 0
    -/
    change (f ≫ s.π) b = 0
    /-
      case val.intro
      X Y : SemiNormedGrp₁
      f : Quiver.Hom X Y
      s : CategoryTheory.Limits.CokernelCofork f
      b : ↑X
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π s) …
    -/
    simp
    -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
    /-
      case val.intro
      X Y : SemiNormedGrp₁
      f : Quiver.Hom X Y
      s : CategoryTheory.Limits.CokernelCofork f
      b : ↑X
      ⊢ Eq (0 b) 0
    -/
    erw [zero_apply]
    /-
      🎉 no goals
    -/
  -- The lift has norm at most one:
  /-
    case property
    X Y : SemiNormedGrp₁
    f : Quiver.Hom X Y
    s : CategoryTheory.Limits.CokernelCofork f
    ⊢ (NormedAddGroupHom.lift (↑f).range ↑(CategoryTheory.Limits.Cofork.π s) ⋯).No …
  -/
  exact NormedAddGroupHom.lift_normNoninc _ _ _ s.π.2
  /-
    🎉 no goals
  -/


instance : HasCokernels SemiNormedGrp₁.{u} where
  has_colimit f :=
    HasColimit.mk
      { cocone := cokernelCocone f
        isColimit :=
          isColimitAux _ (cokernelLift f)
            (fun s => by
              /-
                X✝ Y✝ : SemiNormedGrp₁
                f : Quiver.Hom X✝ Y✝
                s : CategoryTheory.Limits.CokernelCofork f
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp₁.cokernelCocone f).π ( …
              -/
              ext
              /-
                case w.h
                X✝ Y✝ : SemiNormedGrp₁
                f : Quiver.Hom X✝ Y✝
                s : CategoryTheory.Limits.CokernelCofork f
                x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
                ⊢ Eq ((CategoryTheory.CategoryStruct.comp (SemiNormedGrp₁.cokernelCocone f).π  …
              -/
              apply NormedAddGroupHom.lift_mk f.1.range
              /-
                case w.h.hf
                X✝ Y✝ : SemiNormedGrp₁
                f : Quiver.Hom X✝ Y✝
                s : CategoryTheory.Limits.CokernelCofork f
                x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
                ⊢ ∀ (s_1 : ↑Y✝), Membership.mem (↑f).range s_1 → Eq (↑(CategoryTheory.Limits.C …
              -/
              rintro _ ⟨b, rfl⟩
              /-
                case w.h.hf.intro
                X✝ Y✝ : SemiNormedGrp₁
                f : Quiver.Hom X✝ Y✝
                s : CategoryTheory.Limits.CokernelCofork f
                x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
                b : ↑X✝
                ⊢ Eq (↑(CategoryTheory.Limits.Cofork.π s) ((↑f).toAddMonoidHom b)) 0
              -/
              change (f ≫ s.π) b = 0
              /-
                case w.h.hf.intro
                X✝ Y✝ : SemiNormedGrp₁
                f : Quiver.Hom X✝ Y✝
                s : CategoryTheory.Limits.CokernelCofork f
                x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
                b : ↑X✝
                ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π s) …
              -/
              simp
              -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
              /-
                case w.h.hf.intro
                X✝ Y✝ : SemiNormedGrp₁
                f : Quiver.Hom X✝ Y✝
                s : CategoryTheory.Limits.CokernelCofork f
                x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
                b : ↑X✝
                ⊢ Eq (0 b) 0
              -/
              erw [zero_apply])
              /-
                🎉 no goals
              -/
            fun _ _ w =>
            Subtype.eq
              (NormedAddGroupHom.lift_unique f.1.range _ _ _ (congr_arg Subtype.val w : _)) }

-- Sanity check

instance {V W : SemiNormedGrp.{u}} : Sub (V ⟶ W) :=
  (inferInstance : Sub (NormedAddGroupHom V W))

noncomputable instance {V W : SemiNormedGrp.{u}} : Norm (V ⟶ W) :=
  (inferInstance : Norm (NormedAddGroupHom V W))

noncomputable instance {V W : SemiNormedGrp.{u}} : NNNorm (V ⟶ W) :=
  (inferInstance : NNNorm (NormedAddGroupHom V W))

/-- The equalizer cone for a parallel pair of morphisms of seminormed groups. -/
def fork {V W : SemiNormedGrp.{u}} (f g : V ⟶ W) : Fork f g :=
  @Fork.ofι _ _ _ _ _ _ (of (f - g : NormedAddGroupHom V W).ker)
    (NormedAddGroupHom.incl (f - g).ker) <| by
    -- Porting note: not needed in mathlib3
    /-
      V W : SemiNormedGrp
      f g : Quiver.Hom V W
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (NormedAddGroupHom.incl (NormedAddGro …
    -/
    change NormedAddGroupHom V W at f g
    /-
      V W : SemiNormedGrp
      f g : NormedAddGroupHom ↑V ↑W
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (NormedAddGroupHom.incl (NormedAddGro …
    -/
    ext v
    /-
      case h
      V W : SemiNormedGrp
      f g : NormedAddGroupHom ↑V ↑W
      v : ↑(SemiNormedGrp.of (Subtype fun x => Membership.mem (NormedAddGroupHom.ker …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (NormedAddGroupHom.incl (NormedAddGr …
    -/
    have : v.1 ∈ (f - g).ker := v.2
    simpa only [NormedAddGroupHom.incl_apply, Pi.zero_apply, coe_comp, NormedAddGroupHom.coe_zero,
      NormedAddGroupHom.mem_ker, NormedAddGroupHom.coe_sub, Pi.sub_apply,
      sub_eq_zero] using this


instance hasLimit_parallelPair {V W : SemiNormedGrp.{u}} (f g : V ⟶ W) :
    HasLimit (parallelPair f g) where
  exists_limit :=
    Nonempty.intro
      { cone := fork f g
        isLimit :=
          Fork.IsLimit.mk _
            (fun c =>
              NormedAddGroupHom.ker.lift (Fork.ι c) _ <|
                show NormedAddGroupHom.compHom (f - g) c.ι = 0 by
                  /-
                    V W : SemiNormedGrp
                    f g : Quiver.Hom V W
                    c : CategoryTheory.Limits.Fork f g
                    ⊢ Eq ((NormedAddGroupHom.compHom (HSub.hSub f g)) c.ι) 0
                  -/
                  rw [AddMonoidHom.map_sub, AddMonoidHom.sub_apply, sub_eq_zero]; exact c.condition)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
            (fun _ => NormedAddGroupHom.ker.incl_comp_lift _ _ _) fun c g h => by
        -- Porting note: the `simp_rw` was `rw [← h]` but motive is not type correct in mathlib4
              /-
                V W : SemiNormedGrp
                f g✝ : Quiver.Hom V W
                c : CategoryTheory.Limits.Fork f g✝
                g : Quiver.Hom c.pt (SemiNormedGrp.fork f g✝).pt
                h : Eq (CategoryTheory.CategoryStruct.comp g (SemiNormedGrp.fork f g✝).ι) c.ι
                ⊢ Eq g ((fun c => NormedAddGroupHom.ker.lift c.ι (HSub.hSub f g✝) ⋯) c)
              -/
              ext x; dsimp; simp_rw [← h]; rfl}
                                           /-
                                             🎉 no goals
                                           -/


instance : Limits.HasEqualizers.{u, u + 1} SemiNormedGrp :=
  @hasEqualizers_of_hasLimit_parallelPair SemiNormedGrp _ fun {_ _ f g} =>
    SemiNormedGrp.hasLimit_parallelPair f g


/-- Auxiliary definition for `HasCokernels SemiNormedGrp`. -/
noncomputable
def cokernelCocone {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) : Cofork f 0 :=
  @Cofork.ofπ _ _ _ _ _ _ (SemiNormedGrp.of (Y ⧸ NormedAddGroupHom.range f)) f.range.normedMk
    (by
      /-
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (NormedAddGroupHom.range f).normedM …
      -/
      ext a
      /-
        case h
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        a : ↑X
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (NormedAddGroupHom.range f).normed …
      -/
      simp only [comp_apply, Limits.zero_comp]
      -- Porting note: `simp` not firing on the below
      /-
        case h
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        a : ↑X
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (NormedAddGroupHom.range f).normed …
      -/
      rw [comp_apply, NormedAddGroupHom.zero_apply]
      -- Porting note: Lean 3 didn't need this instance
      letI : SeminormedAddCommGroup ((forget SemiNormedGrp).obj Y) :=
        (inferInstance : SeminormedAddCommGroup Y)
      -- Porting note: again simp doesn't seem to be firing in the below line
      /-
        case h
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        a : ↑X
        this : SeminormedAddCommGroup ((CategoryTheory.forget SemiNormedGrp).obj Y) := …
        ⊢ Eq ((NormedAddGroupHom.range f).normedMk (f a)) 0
      -/
      rw [← NormedAddGroupHom.mem_ker, f.range.ker_normedMk, f.mem_range]
    -- This used to be `simp only [exists_apply_eq_apply]` before https://github.com/leanprover/lean4/pull/2644
      /-
        case h
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        a : ↑X
        this : SeminormedAddCommGroup ((CategoryTheory.forget SemiNormedGrp).obj Y) := …
        ⊢ Exists fun w => Eq (f w) (f a)
      -/
      convert exists_apply_eq_apply f a)
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `HasCokernels SemiNormedGrp`. -/
noncomputable
def cokernelLift {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) (s : CokernelCofork f) :
    (cokernelCocone f).pt ⟶ s.pt :=
  NormedAddGroupHom.lift _ s.π
    (by
      /-
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        ⊢ ∀ (s_1 : ↑Y), Membership.mem (NormedAddGroupHom.range f) s_1 → Eq ((Category …
      -/
      rintro _ ⟨b, rfl⟩
      /-
        case intro
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        b : ↑X
        ⊢ Eq ((CategoryTheory.Limits.Cofork.π s) ((NormedAddGroupHom.toAddMonoidHom f) …
      -/
      change (f ≫ s.π) b = 0
      /-
        case intro
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        b : ↑X
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π s) …
      -/
      simp
      -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
      /-
        case intro
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        b : ↑X
        ⊢ Eq (0 b) 0
      -/
      erw [zero_apply])
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `HasCokernels SemiNormedGrp`. -/
noncomputable
def isColimitCokernelCocone {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    IsColimit (cokernelCocone f) :=
  isColimitAux _ (cokernelLift f)
    (fun s => by
      /-
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.cokernelCocone f).π (S …
      -/
      ext
      /-
        case h
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp (SemiNormedGrp.cokernelCocone f).π ( …
      -/
      apply NormedAddGroupHom.lift_mk f.range
      /-
        case h.hf
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
        ⊢ ∀ (s_1 : ↑Y), Membership.mem (NormedAddGroupHom.range f) s_1 → Eq ((Category …
      -/
      rintro _ ⟨b, rfl⟩
      /-
        case h.hf.intro
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
        b : ↑X
        ⊢ Eq ((CategoryTheory.Limits.Cofork.π s) ((NormedAddGroupHom.toAddMonoidHom f) …
      -/
      change (f ≫ s.π) b = 0
      /-
        case h.hf.intro
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
        b : ↑X
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π s) …
      -/
      simp
      -- This used to be the end of the proof before https://github.com/leanprover/lean4/pull/2644
      /-
        case h.hf.intro
        X Y : SemiNormedGrp
        f : Quiver.Hom X Y
        s : CategoryTheory.Limits.CokernelCofork f
        x✝ : ↑((CategoryTheory.Limits.parallelPair f 0).obj CategoryTheory.Limits.Walk …
        b : ↑X
        ⊢ Eq (0 b) 0
      -/
      erw [zero_apply])
      /-
        🎉 no goals
      -/
    fun _ _ w => NormedAddGroupHom.lift_unique f.range _ _ _ w


instance : HasCokernels SemiNormedGrp.{u} where
  has_colimit f :=
    HasColimit.mk
      { cocone := cokernelCocone f
        isColimit := isColimitCokernelCocone f }

-- Sanity check

/-- An explicit choice of cokernel, which has good properties with respect to the norm. -/
noncomputable
def explicitCokernel {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) : SemiNormedGrp.{u} :=
  (cokernelCocone f).pt


/-- Descend to the explicit cokernel. -/
noncomputable
def explicitCokernelDesc {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z} (w : f ≫ g = 0) :
    explicitCokernel f ⟶ Z :=
                                                     /-
                                                       X Y Z : SemiNormedGrp
                                                       f : Quiver.Hom X Y
                                                       g : Quiver.Hom Y Z
                                                       w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct.c …
                                                     -/
  (isColimitCokernelCocone f).desc (Cofork.ofπ g (by simp [w]))
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The projection from `Y` to the explicit cokernel of `X ⟶ Y`. -/
noncomputable
def explicitCokernelπ {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) : Y ⟶ explicitCokernel f :=
  (cokernelCocone f).ι.app WalkingParallelPair.one


theorem explicitCokernelπ_surjective {X Y : SemiNormedGrp.{u}} {f : X ⟶ Y} :
    Function.Surjective (explicitCokernelπ f) :=
  Quot.mk_surjective


@[simp, reassoc]
theorem comp_explicitCokernelπ {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    f ≫ explicitCokernelπ f = 0 := by
  /-
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (SemiNormedGrp.explicitCokernelπ f) …
  -/
  convert (cokernelCocone f).w WalkingParallelPairHom.left
  /-
    case h.e'_3.h
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    e_1✝ : Eq (Quiver.Hom X (SemiNormedGrp.explicitCokernel f)) (Quiver.Hom ((Cate …
    ⊢ Eq 0 ((SemiNormedGrp.cokernelCocone f).ι.app CategoryTheory.Limits.WalkingPa …
  -/
  simp
  /-
    🎉 no goals
  -/

-- Porting note: wasn't necessary in Lean 3. Is this a bug?

@[simp]
theorem explicitCokernelπ_apply_dom_eq_zero {X Y : SemiNormedGrp.{u}} {f : X ⟶ Y} (x : X) :
    (explicitCokernelπ f) (f x) = 0 :=
                                          /-
                                            X Y : SemiNormedGrp
                                            f : Quiver.Hom X Y
                                            x : ↑X
                                            ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (SemiNormedGrp.explicitCokernelπ f …
                                          -/
  show (f ≫ explicitCokernelπ f) x = 0 by rw [comp_explicitCokernelπ]; rfl
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


@[simp, reassoc]
theorem explicitCokernelπ_desc {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    (w : f ≫ g = 0) : explicitCokernelπ f ≫ explicitCokernelDesc w = g :=
  (isColimitCokernelCocone f).fac _ _


@[simp]
theorem explicitCokernelπ_desc_apply {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    {cond : f ≫ g = 0} (x : Y) : explicitCokernelDesc cond (explicitCokernelπ f x) = g x :=
                                                                    /-
                                                                      X Y Z : SemiNormedGrp
                                                                      f : Quiver.Hom X Y
                                                                      g : Quiver.Hom Y Z
                                                                      cond : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                                                      x : ↑Y
                                                                      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f)  …
                                                                    -/
  show (explicitCokernelπ f ≫ explicitCokernelDesc cond) x = g x by rw [explicitCokernelπ_desc]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem explicitCokernelDesc_unique {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    (w : f ≫ g = 0) (e : explicitCokernel f ⟶ Z) (he : explicitCokernelπ f ≫ e = g) :
    e = explicitCokernelDesc w := by
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    e : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    he : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f …
    ⊢ Eq e (SemiNormedGrp.explicitCokernelDesc w)
  -/
  apply (isColimitCokernelCocone f).uniq (Cofork.ofπ g (by simp [w]))
  /-
    case x
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    e : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    he : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f …
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (CategoryTheory.Catego …
  -/
  rintro (_ | _)
    /-
      case x.zero
      X Y Z : SemiNormedGrp
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      e : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
      he : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((SemiNormedGrp.cokernelCocone f).ι.a …
    -/
  · convert w.symm
    /-
      case h.e'_2.h
      X Y Z : SemiNormedGrp
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      e : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
      he : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f …
      e_1✝ : Eq (Quiver.Hom ((CategoryTheory.Limits.parallelPair f 0).obj CategoryTh …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((SemiNormedGrp.cokernelCocone f).ι.a …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case x.one
      X Y Z : SemiNormedGrp
      f : Quiver.Hom X Y
      g : Quiver.Hom Y Z
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      e : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
      he : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((SemiNormedGrp.cokernelCocone f).ι.a …
    -/
  · exact he
    /-
      🎉 no goals
    -/


theorem explicitCokernelDesc_comp_eq_desc {X Y Z W : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    -- Porting note: renamed `cond` to `cond'` to avoid
    -- failed to rewrite using equation theorems for 'cond'
    {h : Z ⟶ W} {cond' : f ≫ g = 0} :
    explicitCokernelDesc cond' ≫ h =
      explicitCokernelDesc
                               /-
                                 X Y Z W : SemiNormedGrp
                                 f : Quiver.Hom X Y
                                 g : Quiver.Hom Y Z
                                 h : Quiver.Hom Z W
                                 cond' : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                               -/
        (show f ≫ g ≫ h = 0 by rw [← CategoryTheory.Category.assoc, cond', Limits.zero_comp]) := by
                               /-
                                 🎉 no goals
                               -/
  /-
    X Y Z W : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z W
    cond' : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelDesc c …
  -/
  refine explicitCokernelDesc_unique _ _ ?_
  /-
    X Y Z W : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z W
    cond' : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) ( …
  -/
  rw [← CategoryTheory.Category.assoc, explicitCokernelπ_desc]
  /-
    🎉 no goals
  -/


@[simp]
theorem explicitCokernelDesc_zero {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} :
    explicitCokernelDesc (show f ≫ (0 : Y ⟶ Z) = 0 from CategoryTheory.Limits.comp_zero) = 0 :=
  Eq.symm <| explicitCokernelDesc_unique _ _ CategoryTheory.Limits.comp_zero


@[ext]
theorem explicitCokernel_hom_ext {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y}
    (e₁ e₂ : explicitCokernel f ⟶ Z) (h : explicitCokernelπ f ≫ e₁ = explicitCokernelπ f ≫ e₂) :
    e₁ = e₂ := by
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    e₁ e₂ : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    h : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    ⊢ Eq e₁ e₂
  -/
  let g : Y ⟶ Z := explicitCokernelπ f ≫ e₂
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    e₁ e₂ : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    h : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    g : Quiver.Hom Y Z := CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explic …
    ⊢ Eq e₁ e₂
  -/
  have w : f ≫ g = 0 := by simp [g]
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    e₁ e₂ : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    h : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    g : Quiver.Hom Y Z := CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explic …
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq e₁ e₂
  -/
  have : e₂ = explicitCokernelDesc w := by apply explicitCokernelDesc_unique; rfl
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    e₁ e₂ : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    h : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    g : Quiver.Hom Y Z := CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explic …
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    this : Eq e₂ (SemiNormedGrp.explicitCokernelDesc w)
    ⊢ Eq e₁ e₂
  -/
  rw [this]
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    e₁ e₂ : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    h : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    g : Quiver.Hom Y Z := CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explic …
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    this : Eq e₂ (SemiNormedGrp.explicitCokernelDesc w)
    ⊢ Eq e₁ (SemiNormedGrp.explicitCokernelDesc w)
  -/
  apply explicitCokernelDesc_unique
  /-
    case he
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    e₁ e₂ : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    h : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    g : Quiver.Hom Y Z := CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explic …
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    this : Eq e₂ (SemiNormedGrp.explicitCokernelDesc w)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) e …
  -/
  exact h
  /-
    🎉 no goals
  -/


instance explicitCokernelπ.epi {X Y : SemiNormedGrp.{u}} {f : X ⟶ Y} :
    Epi (explicitCokernelπ f) := by
  /-
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    ⊢ CategoryTheory.Epi (SemiNormedGrp.explicitCokernelπ f)
  -/
  constructor
  /-
    case left_cancellation
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    ⊢ ∀ {Z : SemiNormedGrp} (g h : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z …
  -/
  intro Z g h H
  /-
    case left_cancellation
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    Z : SemiNormedGrp
    g h : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    H : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    ⊢ Eq g h
  -/
  ext x
  -- Porting note: no longer needed
  -- obtain ⟨x, hx⟩ := explicitCokernelπ_surjective (explicitCokernelπ f x)
  -- change (explicitCokernelπ f ≫ g) _ = _
  /-
    case left_cancellation.h.h
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    Z : SemiNormedGrp
    g h : Quiver.Hom (SemiNormedGrp.explicitCokernel f) Z
    H : Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) …
    x : ↑Y
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f)  …
  -/
  rw [H]
  /-
    🎉 no goals
  -/


theorem isQuotient_explicitCokernelπ {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    NormedAddGroupHom.IsQuotient (explicitCokernelπ f) :=
  NormedAddGroupHom.isQuotientQuotient _


theorem normNoninc_explicitCokernelπ {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    (explicitCokernelπ f).NormNoninc :=
  (isQuotient_explicitCokernelπ f).norm_le


theorem explicitCokernelDesc_norm_le_of_norm_le {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y}
    {g : Y ⟶ Z} (w : f ≫ g = 0) (c : ℝ≥0) (h : ‖g‖ ≤ c) : ‖explicitCokernelDesc w‖ ≤ c :=
  NormedAddGroupHom.lift_norm_le _ _ _ h


theorem explicitCokernelDesc_normNoninc {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    {cond : f ≫ g = 0} (hg : g.NormNoninc) : (explicitCokernelDesc cond).NormNoninc := by
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    cond : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    hg : NormedAddGroupHom.NormNoninc g
    ⊢ NormedAddGroupHom.NormNoninc (SemiNormedGrp.explicitCokernelDesc cond)
  -/
  refine NormedAddGroupHom.NormNoninc.normNoninc_iff_norm_le_one.2 ?_
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    cond : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    hg : NormedAddGroupHom.NormNoninc g
    ⊢ LE.le (Norm.norm (SemiNormedGrp.explicitCokernelDesc cond)) 1
  -/
  rw [← NNReal.coe_one]
  exact
    explicitCokernelDesc_norm_le_of_norm_le cond 1
      (NormedAddGroupHom.NormNoninc.normNoninc_iff_norm_le_one.1 hg)


theorem explicitCokernelDesc_comp_eq_zero {X Y Z W : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    {h : Z ⟶ W} (cond : f ≫ g = 0) (cond2 : g ≫ h = 0) : explicitCokernelDesc cond ≫ h = 0 := by
  /-
    X Y Z W : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z W
    cond : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    cond2 : Eq (CategoryTheory.CategoryStruct.comp g h) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelDesc c …
  -/
  rw [← cancel_epi (explicitCokernelπ f), ← Category.assoc, explicitCokernelπ_desc]
  /-
    X Y Z W : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Quiver.Hom Z W
    cond : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    cond2 : Eq (CategoryTheory.CategoryStruct.comp g h) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g h) (CategoryTheory.CategoryStruct.c …
  -/
  simp [cond2]
  /-
    🎉 no goals
  -/


theorem explicitCokernelDesc_norm_le {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    (w : f ≫ g = 0) : ‖explicitCokernelDesc w‖ ≤ ‖g‖ :=
  explicitCokernelDesc_norm_le_of_norm_le w ‖g‖₊ le_rfl


/-- The explicit cokernel is isomorphic to the usual cokernel. -/
noncomputable
def explicitCokernelIso {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    explicitCokernel f ≅ cokernel f :=
  (isColimitCokernelCocone f).coconePointUniqueUpToIso (colimit.isColimit _)


@[simp]
theorem explicitCokernelIso_hom_π {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    explicitCokernelπ f ≫ (explicitCokernelIso f).hom = cokernel.π _ := by
  /-
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelπ f) ( …
  -/
  simp [explicitCokernelπ, explicitCokernelIso, IsColimit.coconePointUniqueUpToIso]
  /-
    🎉 no goals
  -/


@[simp]
theorem explicitCokernelIso_inv_π {X Y : SemiNormedGrp.{u}} (f : X ⟶ Y) :
    cokernel.π f ≫ (explicitCokernelIso f).inv = explicitCokernelπ f := by
  /-
    X Y : SemiNormedGrp
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π f)  …
  -/
  simp [explicitCokernelπ, explicitCokernelIso]
  /-
    🎉 no goals
  -/


@[simp]
theorem explicitCokernelIso_hom_desc {X Y Z : SemiNormedGrp.{u}} {f : X ⟶ Y} {g : Y ⟶ Z}
    (w : f ≫ g = 0) :
    (explicitCokernelIso f).hom ≫ cokernel.desc f g w = explicitCokernelDesc w := by
  /-
    X Y Z : SemiNormedGrp
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelIso f) …
  -/
  ext1
  simp [explicitCokernelDesc, explicitCokernelπ, explicitCokernelIso,
    IsColimit.coconePointUniqueUpToIso]


/-- A special case of `CategoryTheory.Limits.cokernel.map` adapted to `explicitCokernel`. -/
noncomputable def explicitCokernel.map {A B C D : SemiNormedGrp.{u}}
    {fab : A ⟶ B} {fbd : B ⟶ D} {fac : A ⟶ C} {fcd : C ⟶ D} (h : fab ≫ fbd = fac ≫ fcd) :
    explicitCokernel fab ⟶ explicitCokernel fcd :=
                                                                    /-
                                                                      A B C D : SemiNormedGrp
                                                                      fab : Quiver.Hom A B
                                                                      fbd : Quiver.Hom B D
                                                                      fac : Quiver.Hom A C
                                                                      fcd : Quiver.Hom C D
                                                                      h : Eq (CategoryTheory.CategoryStruct.comp fab fbd) (CategoryTheory.CategorySt …
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp fab (CategoryTheory.CategoryStruct.co …
                                                                    -/
  @explicitCokernelDesc _ _ _ fab (fbd ≫ explicitCokernelπ _) <| by simp [reassoc_of% h]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- A special case of `CategoryTheory.Limits.cokernel.map_desc` adapted to `explicitCokernel`. -/
theorem ExplicitCoker.map_desc {A B C D B' D' : SemiNormedGrp.{u}}
    {fab : A ⟶ B} {fbd : B ⟶ D} {fac : A ⟶ C} {fcd : C ⟶ D} {h : fab ≫ fbd = fac ≫ fcd}
    {fbb' : B ⟶ B'} {fdd' : D ⟶ D'} {condb : fab ≫ fbb' = 0} {condd : fcd ≫ fdd' = 0} {g : B' ⟶ D'}
    (h' : fbb' ≫ g = fbd ≫ fdd') :
    explicitCokernelDesc condb ≫ g = explicitCokernel.map h ≫ explicitCokernelDesc condd := by
  /-
    A B C D B' D' : SemiNormedGrp
    fab : Quiver.Hom A B
    fbd : Quiver.Hom B D
    fac : Quiver.Hom A C
    fcd : Quiver.Hom C D
    h : Eq (CategoryTheory.CategoryStruct.comp fab fbd) (CategoryTheory.CategorySt …
    fbb' : Quiver.Hom B B'
    fdd' : Quiver.Hom D D'
    condb : Eq (CategoryTheory.CategoryStruct.comp fab fbb') 0
    condd : Eq (CategoryTheory.CategoryStruct.comp fcd fdd') 0
    g : Quiver.Hom B' D'
    h' : Eq (CategoryTheory.CategoryStruct.comp fbb' g) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelDesc c …
  -/
  delta explicitCokernel.map
  /-
    A B C D B' D' : SemiNormedGrp
    fab : Quiver.Hom A B
    fbd : Quiver.Hom B D
    fac : Quiver.Hom A C
    fcd : Quiver.Hom C D
    h : Eq (CategoryTheory.CategoryStruct.comp fab fbd) (CategoryTheory.CategorySt …
    fbb' : Quiver.Hom B B'
    fdd' : Quiver.Hom D D'
    condb : Eq (CategoryTheory.CategoryStruct.comp fab fbb') 0
    condd : Eq (CategoryTheory.CategoryStruct.comp fcd fdd') 0
    g : Quiver.Hom B' D'
    h' : Eq (CategoryTheory.CategoryStruct.comp fbb' g) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SemiNormedGrp.explicitCokernelDesc c …
  -/
  simp only [← Category.assoc, ← cancel_epi (explicitCokernelπ fab)]
  /-
    A B C D B' D' : SemiNormedGrp
    fab : Quiver.Hom A B
    fbd : Quiver.Hom B D
    fac : Quiver.Hom A C
    fcd : Quiver.Hom C D
    h : Eq (CategoryTheory.CategoryStruct.comp fab fbd) (CategoryTheory.CategorySt …
    fbb' : Quiver.Hom B B'
    fdd' : Quiver.Hom D D'
    condb : Eq (CategoryTheory.CategoryStruct.comp fab fbb') 0
    condd : Eq (CategoryTheory.CategoryStruct.comp fcd fdd') 0
    g : Quiver.Hom B' D'
    h' : Eq (CategoryTheory.CategoryStruct.comp fbb' g) (CategoryTheory.CategorySt …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [Category.assoc, explicitCokernelπ_desc, h']
  /-
    🎉 no goals
  -/


