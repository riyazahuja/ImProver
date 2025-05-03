instance semiringObj (j : J) :
    Semiring (((F ⋙ forget₂ SemiRingCatMax.{v, u} MonCat) ⋙ forget MonCat).obj j) :=
                             /-
                               J : Type v
                               inst✝ : CategoryTheory.SmallCategory J
                               F : CategoryTheory.Functor J SemiRingCatMax
                               j : J
                               ⊢ Semiring ↑(F.obj j)
                             -/
  show Semiring (F.obj j) by infer_instance
                             /-
                               🎉 no goals
                             -/


/-- The colimit of `F ⋙ forget₂ SemiRingCat MonCat` in the category `MonCat`.
In the following, we will show that this has the structure of a semiring.
-/
abbrev R : MonCatMax.{v, u} :=
  MonCat.FilteredColimits.colimit.{v, u} (F ⋙ forget₂ SemiRingCatMax.{v, u} MonCatMax.{v, u})


instance colimitSemiring : Semiring.{max v u} <| R.{v, u} F :=
  { (R.{v, u} F).str,
    AddCommMonCat.FilteredColimits.colimitAddCommMonoid.{v, u}
      (F ⋙ forget₂ SemiRingCat AddCommMonCat.{max v u}) with
    mul_zero := fun x => by
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : ↑(SemiRingCat.FilteredColimits.R F)
        ⊢ Eq (HMul.hMul x 0) 0
      -/
      refine Quot.inductionOn x ?_; clear x; intro x
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : Sigma fun j => ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)) …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      obtain ⟨j, x⟩ := x
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      erw [colimit_zero_eq _ j, colimit_mul_mk_eq _ ⟨j, _⟩ ⟨j, _⟩ j (𝟙 j) (𝟙 j)]
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      rw [CategoryTheory.Functor.map_id]
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : ↑(SemiRingCat.FilteredColimits.R F)
        ⊢ Eq (HMul.hMul 0 x) 0
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x : Sigma fun j => ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)) …
        ⊢ Eq (HMul.hMul 0 (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Cat …
      -/
      dsimp
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (HMul.hMul 0 (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Cat …
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      rw [mul_zero x]
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : ↑(SemiRingCat.FilteredColimits.R F)
        ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (HAdd.hAdd (HMul.hMul x y) (HMul.hMul x z))
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : Sigma fun j => ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatM …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      rfl
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      /-
        case mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (HMul.hMul (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F.comp (Categ …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
    zero_mul := fun x => by
      refine Quot.inductionOn x ?_; clear x; intro x
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        h : Quiver.Hom j₃ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      obtain ⟨j, x⟩ := x
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        h : Quiver.Hom j₃ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      erw [colimit_zero_eq _ j, colimit_mul_mk_eq _ ⟨j, _⟩ ⟨j, _⟩ j (𝟙 j) (𝟙 j)]
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        h : Quiver.Hom j₃ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      rw [CategoryTheory.Functor.map_id]
      /-
        🎉 no goals
      -/
      dsimp
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : ↑(SemiRingCat.FilteredColimits.R F)
        ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))
      -/
      rw [zero_mul x]
      /-
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        x y z : Sigma fun j => ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatM …
        ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F. …
      -/
      rfl
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F. …
      -/
    left_distrib := fun x y z => by
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F. …
      -/
      refine Quot.induction_on₃ x y z ?_; clear x y z; intro x y z
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F. …
      -/
      obtain ⟨j₁, x⟩ := x; obtain ⟨j₂, y⟩ := y; obtain ⟨j₃, z⟩ := z
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (HMul.hMul (HAdd.hAdd (Quot.mk (CategoryTheory.Limits.Types.Quot.Rel ((F. …
      -/
      let k := IsFiltered.max₃ j₁ j₂ j₃
      let f := IsFiltered.firstToMax₃ j₁ j₂ j₃
      let g := IsFiltered.secondToMax₃ j₁ j₂ j₃
      let h := IsFiltered.thirdToMax₃ j₁ j₂ j₃
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        h : Quiver.Hom j₃ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
      erw [colimit_add_mk_eq _ ⟨j₂, _⟩ ⟨j₃, _⟩ k g h, colimit_mul_mk_eq _ ⟨j₁, _⟩ ⟨k, _⟩ k f (𝟙 k),
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        h : Quiver.Hom j₃ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
        colimit_mul_mk_eq _ ⟨j₁, _⟩ ⟨j₂, _⟩ k f g, colimit_mul_mk_eq _ ⟨j₁, _⟩ ⟨j₃, _⟩ k f h,
      /-
        case mk.mk.mk
        J : Type v
        inst✝¹ : CategoryTheory.SmallCategory J
        F : CategoryTheory.Functor J SemiRingCatMax
        inst✝ : CategoryTheory.IsFiltered J
        j₁ : J
        x : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₂ : J
        y : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        j₃ : J
        z : ((F.comp (CategoryTheory.forget₂ SemiRingCatMax MonCatMax)).comp (Category …
        k : J := CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃
        f : Quiver.Hom j₁ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        g : Quiver.Hom j₂ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        h : Quiver.Hom j₃ (CategoryTheory.IsFiltered.max₃ j₁ j₂ j₃) := CategoryTheory. …
        ⊢ Eq (MonCat.FilteredColimits.M.mk (F.comp (CategoryTheory.forget₂ SemiRingCat …
      -/
        colimit_add_mk_eq _ ⟨k, _⟩ ⟨k, _⟩ k (𝟙 k) (𝟙 k)]
      /-
        🎉 no goals
      -/
      simp only [CategoryTheory.Functor.map_id, id_apply]
      erw [left_distrib (F.map f x) (F.map g y) (F.map h z)]
      rfl
    right_distrib := fun x y z => by
      refine Quot.induction_on₃ x y z ?_; clear x y z; intro x y z
      obtain ⟨j₁, x⟩ := x; obtain ⟨j₂, y⟩ := y; obtain ⟨j₃, z⟩ := z
      let k := IsFiltered.max₃ j₁ j₂ j₃
      let f := IsFiltered.firstToMax₃ j₁ j₂ j₃
      let g := IsFiltered.secondToMax₃ j₁ j₂ j₃
      let h := IsFiltered.thirdToMax₃ j₁ j₂ j₃
      erw [colimit_add_mk_eq _ ⟨j₁, _⟩ ⟨j₂, _⟩ k f g, colimit_mul_mk_eq _ ⟨k, _⟩ ⟨j₃, _⟩ k (𝟙 k) h,
        colimit_mul_mk_eq _ ⟨j₁, _⟩ ⟨j₃, _⟩ k f h, colimit_mul_mk_eq _ ⟨j₂, _⟩ ⟨j₃, _⟩ k g h,
        colimit_add_mk_eq _ ⟨k, _⟩ ⟨k, _⟩ k (𝟙 k) (𝟙 k)]
      simp only [CategoryTheory.Functor.map_id, id_apply]
      erw [right_distrib (F.map f x) (F.map g y) (F.map h z)]
      rfl }


/-- The bundled semiring giving the filtered colimit of a diagram. -/
def colimit : SemiRingCatMax.{v, u} :=
  SemiRingCat.of <| R.{v, u} F


/-- The cocone over the proposed colimit semiring. -/
def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι :=
    { app := fun j => ofHom
        { (MonCat.FilteredColimits.colimitCocone
            (F ⋙ forget₂ SemiRingCatMax.{v, u} MonCat)).ι.app j,
            (AddCommMonCat.FilteredColimits.colimitCocone
              (F ⋙ forget₂ SemiRingCatMax.{v, u} AddCommMonCat)).ι.app j with }
      naturality := fun {_ _} f => hom_ext <|
        RingHom.coe_inj ((Types.TypeMax.colimitCocone (F ⋙ forget SemiRingCat)).ι.naturality f) }


/-- The proposed colimit cocone is a colimit in `SemiRingCat`. -/
def colimitCoconeIsColimit : IsColimit <| colimitCocone.{v, u} F where
  desc t := ofHom
    { (MonCat.FilteredColimits.colimitCoconeIsColimit.{v, u}
            (F ⋙ forget₂ SemiRingCatMax.{v, u} MonCat)).desc
        ((forget₂ SemiRingCatMax.{v, u} MonCat).mapCocone t),
      (AddCommMonCat.FilteredColimits.colimitCoconeIsColimit.{v, u}
            (F ⋙ forget₂ SemiRingCatMax.{v, u} AddCommMonCat)).desc
        ((forget₂ SemiRingCatMax.{v, u} AddCommMonCat).mapCocone t) with }
  fac t j := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget SemiRingCatMax.{v, u})).fac
        ((forget SemiRingCatMax.{v, u}).mapCocone t) j
  uniq t _ h := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget SemiRingCat)).uniq
        ((forget SemiRingCat).mapCocone t) _ fun j => funext fun x =>
        RingHom.congr_fun (congrArg Hom.hom (h j)) x


instance forget₂Mon_preservesFilteredColimits :
    PreservesFilteredColimits (forget₂ SemiRingCat MonCat.{u}) where
  preserves_filtered_colimits {J hJ1 _} :=
    letI : Category J := hJ1
    { preservesColimit := fun {F} =>
        preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
          (MonCat.FilteredColimits.colimitCoconeIsColimit (F ⋙ forget₂ SemiRingCat MonCat.{u})) }


instance forget_preservesFilteredColimits : PreservesFilteredColimits (forget SemiRingCat.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ SemiRingCat MonCat) (forget MonCat.{u})


/-- The colimit of `F ⋙ forget₂ CommSemiRingCat SemiRingCat` in the category `SemiRingCat`.
In the following, we will show that this has the structure of a _commutative_ semiring.
-/
abbrev R : SemiRingCatMax.{v, u} :=
  SemiRingCat.FilteredColimits.colimit (F ⋙ forget₂ CommSemiRingCat SemiRingCat.{max v u})


instance colimitCommSemiring : CommSemiring.{max v u} <| R.{v, u} F :=
  { (R F).semiring,
    CommMonCat.FilteredColimits.colimitCommMonoid
      (F ⋙ forget₂ CommSemiRingCat CommMonCat.{max v u}) with }


/-- The bundled commutative semiring giving the filtered colimit of a diagram. -/
def colimit : CommSemiRingCat.{max v u} :=
  CommSemiRingCat.of <| R.{v, u} F


/-- The cocone over the proposed colimit commutative semiring. -/
def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι :=
    { app := fun X ↦ ofHom <| ((SemiRingCat.FilteredColimits.colimitCocone
          (F ⋙ forget₂ CommSemiRingCat SemiRingCat.{max v u})).ι.app X).hom
      naturality := fun _ _ f ↦ hom_ext <|
        RingHom.coe_inj ((Types.TypeMax.colimitCocone
          (F ⋙ forget CommSemiRingCat)).ι.naturality f) }


/-- The proposed colimit cocone is a colimit in `CommSemiRingCat`. -/
def colimitCoconeIsColimit : IsColimit <| colimitCocone.{v, u} F where
  desc t := ofHom <|
    (SemiRingCat.FilteredColimits.colimitCoconeIsColimit.{v, u}
          (F ⋙ forget₂ CommSemiRingCat SemiRingCat.{max v u})).desc
      ((forget₂ CommSemiRingCat SemiRingCat).mapCocone t) |>.hom
  fac t j := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget CommSemiRingCat)).fac
        ((forget CommSemiRingCat).mapCocone t) j
  uniq t _ h := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget CommSemiRingCat)).uniq
        ((forget CommSemiRingCat).mapCocone t) _ fun j => funext fun x =>
        RingHom.congr_fun (congrArg Hom.hom (h j)) x


instance forget₂SemiRing_preservesFilteredColimits :
    PreservesFilteredColimits (forget₂ CommSemiRingCat SemiRingCat.{u}) where
  preserves_filtered_colimits {J hJ1 _} :=
    letI : Category J := hJ1
    { preservesColimit := fun {F} =>
        preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
          (SemiRingCat.FilteredColimits.colimitCoconeIsColimit
            (F ⋙ forget₂ CommSemiRingCat SemiRingCat.{u})) }


instance forget_preservesFilteredColimits :
    PreservesFilteredColimits (forget CommSemiRingCat.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ CommSemiRingCat SemiRingCat)
    (forget SemiRingCat.{u})


/-- The colimit of `F ⋙ forget₂ RingCat SemiRingCat` in the category `SemiRingCat`.
In the following, we will show that this has the structure of a ring.
-/
abbrev R : SemiRingCat.{max v u} :=
  SemiRingCat.FilteredColimits.colimit.{v, u} (F ⋙ forget₂ RingCat SemiRingCat.{max v u})


instance colimitRing : Ring.{max v u} <| R.{v, u} F :=
  { (R F).semiring,
    AddCommGrp.FilteredColimits.colimitAddCommGroup.{v, u}
      (F ⋙ forget₂ RingCat AddCommGrp.{max v u}) with }


/-- The bundled ring giving the filtered colimit of a diagram. -/
def colimit : RingCat.{max v u} :=
  RingCat.of <| R.{v, u} F


/-- The cocone over the proposed colimit ring. -/
def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι :=
    { app := fun X ↦ ofHom <| ((SemiRingCat.FilteredColimits.colimitCocone
          (F ⋙ forget₂ RingCat SemiRingCat.{max v u})).ι.app X).hom
      naturality := fun _ _ f ↦ hom_ext <|
        RingHom.coe_inj ((Types.TypeMax.colimitCocone (F ⋙ forget RingCat)).ι.naturality f) }


/-- The proposed colimit cocone is a colimit in `Ring`. -/
def colimitCoconeIsColimit : IsColimit <| colimitCocone.{v, u} F where
  desc t := ofHom <|
    (SemiRingCat.FilteredColimits.colimitCoconeIsColimit.{v, u}
          (F ⋙ forget₂ RingCat SemiRingCat.{max v u})).desc
      ((forget₂ RingCat SemiRingCat).mapCocone t) |>.hom
  fac t j := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget RingCat)).fac
        ((forget RingCat).mapCocone t) j
  uniq t _ h := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget RingCat)).uniq
        ((forget RingCat).mapCocone t) _ fun j => funext fun x =>
        RingHom.congr_fun (congrArg Hom.hom (h j)) x


instance forget₂SemiRing_preservesFilteredColimits :
    PreservesFilteredColimits (forget₂ RingCat SemiRingCat.{u}) where
  preserves_filtered_colimits {J hJ1 _} :=
    letI : Category J := hJ1
    { preservesColimit := fun {F} =>
        preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
          (SemiRingCat.FilteredColimits.colimitCoconeIsColimit
            (F ⋙ forget₂ RingCat SemiRingCat.{u})) }


instance forget_preservesFilteredColimits : PreservesFilteredColimits (forget RingCat.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ RingCat SemiRingCat) (forget SemiRingCat.{u})


/-- The colimit of `F ⋙ forget₂ CommRingCat RingCat` in the category `RingCat`.
In the following, we will show that this has the structure of a _commutative_ ring.
-/
abbrev R : RingCat.{max v u} :=
  RingCat.FilteredColimits.colimit.{v, u} (F ⋙ forget₂ CommRingCat RingCat.{max v u})


instance colimitCommRing : CommRing.{max v u} <| R.{v, u} F :=
  { (R.{v, u} F).ring,
    CommSemiRingCat.FilteredColimits.colimitCommSemiring
      (F ⋙ forget₂ CommRingCat CommSemiRingCat.{max v u}) with }


/-- The bundled commutative ring giving the filtered colimit of a diagram. -/
def colimit : CommRingCat.{max v u} :=
  CommRingCat.of <| R.{v, u} F


/-- The cocone over the proposed colimit commutative ring. -/
def colimitCocone : Cocone F where
  pt := colimit.{v, u} F
  ι :=
    { app := fun X ↦ ofHom <| ((RingCat.FilteredColimits.colimitCocone
          (F ⋙ forget₂ CommRingCat RingCat.{max v u})).ι.app X).hom
      naturality := fun _ _ f ↦ hom_ext <|
        RingHom.coe_inj ((Types.TypeMax.colimitCocone (F ⋙ forget CommRingCat)).ι.naturality f) }


/-- The proposed colimit cocone is a colimit in `CommRingCat`. -/
def colimitCoconeIsColimit : IsColimit <| colimitCocone.{v, u} F where
  desc t := ofHom <|
    (RingCat.FilteredColimits.colimitCoconeIsColimit.{v, u}
          (F ⋙ forget₂ CommRingCat RingCat.{max v u})).desc
      ((forget₂ CommRingCat RingCat).mapCocone t) |>.hom
  fac t j := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit.{v, u} (F ⋙ forget CommRingCat)).fac
        ((forget CommRingCat).mapCocone t) j
  uniq t _ h := hom_ext <|
    RingHom.coe_inj <|
      (Types.TypeMax.colimitCoconeIsColimit (F ⋙ forget CommRingCat)).uniq
        ((forget CommRingCat).mapCocone t) _ fun j => funext fun x =>
        RingHom.congr_fun (congrArg Hom.hom <| h j) x


instance forget₂Ring_preservesFilteredColimits :
    PreservesFilteredColimits (forget₂ CommRingCat RingCat.{u}) where
  preserves_filtered_colimits {J hJ1 _} :=
    letI : Category J := hJ1
    { preservesColimit := fun {F} =>
        preservesColimit_of_preserves_colimit_cocone (colimitCoconeIsColimit.{u, u} F)
          (RingCat.FilteredColimits.colimitCoconeIsColimit (F ⋙ forget₂ CommRingCat RingCat.{u})) }


instance forget_preservesFilteredColimits : PreservesFilteredColimits (forget CommRingCat.{u}) :=
  Limits.comp_preservesFilteredColimits (forget₂ CommRingCat RingCat) (forget RingCat.{u})


