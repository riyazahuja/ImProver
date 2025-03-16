/-- The left adjoint of evaluation. -/
@[simps]
def evaluationLeftAdjoint (c : C) : D ⥤ C ⥤ D where
  obj d :=
    { obj := fun t => ∐ fun _ : c ⟶ t => d
      map := fun f => Sigma.desc fun g => (Sigma.ι fun _ => d) <| g ≫ f}
  map {_ d₂} f :=
    { app := fun _ => Sigma.desc fun h => f ≫ Sigma.ι (fun _ => d₂) h
      naturality := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
          c : C
          x✝ d₂ : D
          f : Quiver.Hom x✝ d₂
          ⊢ ∀ ⦃X Y : C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
          c : C
          x✝ d₂ : D
          f : Quiver.Hom x✝ d₂
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun d => { obj := fun t => Categor …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
          c : C
          x✝ d₂ : D
          f : Quiver.Hom x✝ d₂
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
          c : C
          x✝ d₂ : D
          f : Quiver.Hom x✝ d₂
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          b✝ : Quiver.Hom c X✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun x …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The adjunction showing that evaluation is a right adjoint. -/
@[simps! unit_app counit_app_app]
def evaluationAdjunctionRight (c : C) : evaluationLeftAdjoint D c ⊣ (evaluation _ _).obj c :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun d F =>
        { toFun := fun f => Sigma.ι (fun _ => d) (𝟙 _) ≫ f.app c
          invFun := fun f =>
            { app := fun _ => Sigma.desc fun h => f ≫ F.map h
              naturality := by
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
                  c : C
                  d : D
                  F : CategoryTheory.Functor C D
                  f : Quiver.Hom d (((CategoryTheory.evaluation C D).obj c).obj F)
                  ⊢ ∀ ⦃X Y : C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
                -/
                intros
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
                  c : C
                  d : D
                  F : CategoryTheory.Functor C D
                  f : Quiver.Hom d (((CategoryTheory.evaluation C D).obj c).obj F)
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.evaluationLeftAdjoi …
                -/
                dsimp
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
                  c : C
                  d : D
                  F : CategoryTheory.Functor C D
                  f : Quiver.Hom d (((CategoryTheory.evaluation C D).obj c).obj F)
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc fun …
                -/
                ext
                /-
                  case h
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
                  c : C
                  d : D
                  F : CategoryTheory.Functor C D
                  f : Quiver.Hom d (((CategoryTheory.evaluation C D).obj c).obj F)
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  b✝ : Quiver.Hom c X✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun x …
                -/
                simp }
                /-
                  🎉 no goals
                -/
          left_inv := by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
              c : C
              d : D
              F : CategoryTheory.Functor C D
              ⊢ Function.LeftInverse (fun f => { app := fun x => CategoryTheory.Limits.Sigma …
            -/
            intro f
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
              c : C
              d : D
              F : CategoryTheory.Functor C D
              f : Quiver.Hom ((CategoryTheory.evaluationLeftAdjoint D c).obj d) F
              ⊢ Eq ((fun f => { app := fun x => CategoryTheory.Limits.Sigma.desc fun h => Ca …
            -/
            ext x
            /-
              case w.h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
              c : C
              d : D
              F : CategoryTheory.Functor C D
              f : Quiver.Hom ((CategoryTheory.evaluationLeftAdjoint D c).obj d) F
              x : C
              ⊢ Eq (((fun f => { app := fun x => CategoryTheory.Limits.Sigma.desc fun h => C …
            -/
            dsimp
            /-
              case w.h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
              c : C
              d : D
              F : CategoryTheory.Functor C D
              f : Quiver.Hom ((CategoryTheory.evaluationLeftAdjoint D c).obj d) F
              x : C
              ⊢ Eq (CategoryTheory.Limits.Sigma.desc fun h => CategoryTheory.CategoryStruct. …
            -/
            ext g
            simp only [colimit.ι_desc, Cofan.mk_ι_app, Category.assoc, ← f.naturality,
              evaluationLeftAdjoint_obj_map, colimit.ι_desc_assoc,
              Discrete.functor_obj, Cofan.mk_pt, Discrete.natTrans_app, Category.id_comp]
          right_inv := fun f => by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
              c : C
              d : D
              F : CategoryTheory.Functor C D
              f : Quiver.Hom d (((CategoryTheory.evaluation C D).obj c).obj F)
              ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigm …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
              c : C
              d : D
              F : CategoryTheory.Functor C D
              f : Quiver.Hom d (((CategoryTheory.evaluation C D).obj c).obj F)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun x …
            -/
            simp }
            /-
              🎉 no goals
            -/
      -- This used to be automatic before https://github.com/leanprover/lean4/pull/2644
                                      /-
                                        C : Type u₁
                                        inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                        D : Type u₂
                                        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                        inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
                                        c : C
                                        ⊢ ∀ {X : D} {Y Y' : CategoryTheory.Functor C D} (f : Quiver.Hom ((CategoryTheo …
                                      -/
      homEquiv_naturality_right := by intros; dsimp; simp }
                                                     /-
                                                       🎉 no goals
                                                     -/


instance evaluationIsRightAdjoint (c : C) : ((evaluation _ D).obj c).IsRightAdjoint  :=
  ⟨_, ⟨evaluationAdjunctionRight _ _⟩⟩


/-- See also the file `CategoryTheory.Limits.FunctorCategory.EpiMono`
for a similar result under a `HasPullbacks` assumption. -/
theorem NatTrans.mono_iff_mono_app' {F G : C ⥤ D} (η : F ⟶ G) : Mono η ↔ ∀ c, Mono (η.app c) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
    F G : CategoryTheory.Functor C D
    η : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Mono η) (∀ (c : C), CategoryTheory.Mono (η.app c))
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      ⊢ CategoryTheory.Mono η → ∀ (c : C), CategoryTheory.Mono (η.app c)
    -/
  · intro h c
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      h : CategoryTheory.Mono η
      c : C
      ⊢ CategoryTheory.Mono (η.app c)
    -/
    exact (inferInstance : Mono (((evaluation _ _).obj c).map η))
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      ⊢ (∀ (c : C), CategoryTheory.Mono (η.app c)) → CategoryTheory.Mono η
    -/
  · intro _
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasCoproductsOfShape (Quiver.Hom a  …
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      a✝ : ∀ (c : C), CategoryTheory.Mono (η.app c)
      ⊢ CategoryTheory.Mono η
    -/
    apply NatTrans.mono_of_mono_app
    /-
      🎉 no goals
    -/


/-- The right adjoint of evaluation. -/
@[simps]
def evaluationRightAdjoint (c : C) : D ⥤ C ⥤ D where
  obj d :=
    { obj := fun t => ∏ᶜ fun _ : t ⟶ c => d
      map := fun f => Pi.lift fun g => Pi.π _ <| f ≫ g }
  map f :=
    { app := fun _ => Pi.lift fun g => Pi.π _ g ≫ f
      naturality := by
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
          c : C
          X✝ Y✝ : D
          f : Quiver.Hom X✝ Y✝
          ⊢ ∀ ⦃X Y : C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
        -/
        intros
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
          c : C
          X✝¹ Y✝¹ : D
          f : Quiver.Hom X✝¹ Y✝¹
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun d => { obj := fun t => Categor …
        -/
        dsimp
        /-
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
          c : C
          X✝¹ Y✝¹ : D
          f : Quiver.Hom X✝¹ Y✝¹
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.lift fun g  …
        -/
        ext
        /-
          case h
          C : Type u₁
          inst✝² : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
          inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
          c : C
          X✝¹ Y✝¹ : D
          f : Quiver.Hom X✝¹ Y✝¹
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          b✝ : Quiver.Hom Y✝ c
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp }
        /-
          🎉 no goals
        -/


/-- The adjunction showing that evaluation is a left adjoint. -/
@[simps! unit_app_app counit_app]
def evaluationAdjunctionLeft (c : C) : (evaluation _ _).obj c ⊣ evaluationRightAdjoint D c :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun F d =>
        { toFun := fun f =>
            { app := fun _ => Pi.lift fun g => F.map g ≫ f
              naturality := by
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
                  c : C
                  F : CategoryTheory.Functor C D
                  d : D
                  f : Quiver.Hom (((CategoryTheory.evaluation C D).obj c).obj F) d
                  ⊢ ∀ ⦃X Y : C⦄ (f_1 : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ( …
                -/
                intros
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
                  c : C
                  F : CategoryTheory.Functor C D
                  d : D
                  f : Quiver.Hom (((CategoryTheory.evaluation C D).obj c).obj F) d
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f✝) ((fun x => CategoryTheory. …
                -/
                dsimp
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
                  c : C
                  F : CategoryTheory.Functor C D
                  d : D
                  f : Quiver.Hom (((CategoryTheory.evaluation C D).obj c).obj F) d
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f✝) (CategoryTheory.Limits.Pi. …
                -/
                ext
                /-
                  case h
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
                  c : C
                  F : CategoryTheory.Functor C D
                  d : D
                  f : Quiver.Hom (((CategoryTheory.evaluation C D).obj c).obj F) d
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  b✝ : Quiver.Hom Y✝ c
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                -/
                simp }
                /-
                  🎉 no goals
                -/
          invFun := fun f => f.app _ ≫ Pi.π _ (𝟙 _)
          left_inv := fun f => by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
              c : C
              F : CategoryTheory.Functor C D
              d : D
              f : Quiver.Hom (((CategoryTheory.evaluation C D).obj c).obj F) d
              ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (f.app c) (CategoryTheory.L …
            -/
            dsimp
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
              c : C
              F : CategoryTheory.Functor C D
              d : D
              f : Quiver.Hom (((CategoryTheory.evaluation C D).obj c).obj F) d
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.lift fun g  …
            -/
            simp
            /-
              🎉 no goals
            -/
          right_inv := by
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
              c : C
              F : CategoryTheory.Functor C D
              d : D
              ⊢ Function.RightInverse (fun f => CategoryTheory.CategoryStruct.comp (f.app c) …
            -/
            intro f
            /-
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
              c : C
              F : CategoryTheory.Functor C D
              d : D
              f : Quiver.Hom F ((CategoryTheory.evaluationRightAdjoint D c).obj d)
              ⊢ Eq ((fun f => { app := fun x => CategoryTheory.Limits.Pi.lift fun g => Categ …
            -/
            ext x
            /-
              case w.h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
              c : C
              F : CategoryTheory.Functor C D
              d : D
              f : Quiver.Hom F ((CategoryTheory.evaluationRightAdjoint D c).obj d)
              x : C
              ⊢ Eq (((fun f => { app := fun x => CategoryTheory.Limits.Pi.lift fun g => Cate …
            -/
            dsimp
            /-
              case w.h
              C : Type u₁
              inst✝² : CategoryTheory.Category.{v₁, u₁} C
              D : Type u₂
              inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
              inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
              c : C
              F : CategoryTheory.Functor C D
              d : D
              f : Quiver.Hom F ((CategoryTheory.evaluationRightAdjoint D c).obj d)
              x : C
              ⊢ Eq (CategoryTheory.Limits.Pi.lift fun g => CategoryTheory.CategoryStruct.com …
            -/
            ext g
            simp only [Discrete.functor_obj, NatTrans.naturality_assoc,
              evaluationRightAdjoint_obj_obj, evaluationRightAdjoint_obj_map, limit.lift_π,
              Fan.mk_pt, Fan.mk_π_app, Discrete.natTrans_app, Category.comp_id] } }


instance evaluationIsLeftAdjoint (c : C) : ((evaluation _ D).obj c).IsLeftAdjoint :=
  ⟨_, ⟨evaluationAdjunctionLeft _ _⟩⟩


/-- See also the file `CategoryTheory.Limits.FunctorCategory.EpiMono`
for a similar result under a `HasPushouts` assumption. -/
theorem NatTrans.epi_iff_epi_app' {F G : C ⥤ D} (η : F ⟶ G) : Epi η ↔ ∀ c, Epi (η.app c) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
    F G : CategoryTheory.Functor C D
    η : Quiver.Hom F G
    ⊢ Iff (CategoryTheory.Epi η) (∀ (c : C), CategoryTheory.Epi (η.app c))
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      ⊢ CategoryTheory.Epi η → ∀ (c : C), CategoryTheory.Epi (η.app c)
    -/
  · intro h c
    /-
      case mp
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      h : CategoryTheory.Epi η
      c : C
      ⊢ CategoryTheory.Epi (η.app c)
    -/
    exact (inferInstance : Epi (((evaluation _ _).obj c).map η))
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      ⊢ (∀ (c : C), CategoryTheory.Epi (η.app c)) → CategoryTheory.Epi η
    -/
  · intros
    /-
      case mpr
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      inst✝ : ∀ (a b : C), CategoryTheory.Limits.HasProductsOfShape (Quiver.Hom a b) D
      F G : CategoryTheory.Functor C D
      η : Quiver.Hom F G
      a✝ : ∀ (c : C), CategoryTheory.Epi (η.app c)
      ⊢ CategoryTheory.Epi η
    -/
    apply NatTrans.epi_of_epi_app
    /-
      🎉 no goals
    -/


