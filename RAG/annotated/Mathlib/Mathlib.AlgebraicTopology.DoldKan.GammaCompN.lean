/-- The isomorphism `(Γ₀.splitting K).nondegComplex ≅ K` for all `K : ChainComplex C ℕ`. -/
@[simps!]
def Γ₀NondegComplexIso (K : ChainComplex C ℕ) : (Γ₀.splitting K).nondegComplex ≅ K :=
  HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _)
    (by
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.43, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        ⊢ ∀ (i j : Nat), (ComplexShape.down Nat).Rel i j → Eq (CategoryTheory.Category …
      -/
      rintro _ n (rfl : n + 1 = _)
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.43, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        n : Nat
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => CategoryTheory.Iso.refl (( …
      -/
      dsimp
      simp only [id_comp, comp_id, AlternatingFaceMapComplex.obj_d_eq, Preadditive.sum_comp,
        Preadditive.comp_sum]
      /-
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.43, u_1} C
        inst✝¹ : CategoryTheory.Preadditive C
        inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
        K : ChainComplex C Nat
        n : Nat
        ⊢ Eq (K.d (HAdd.hAdd n 1) n) (Finset.univ.sum fun j => CategoryTheory.Category …
      -/
      rw [Fintype.sum_eq_single (0 : Fin (n + 2))]
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.43, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K : ChainComplex C Nat
          n : Nat
          ⊢ Eq (K.d (HAdd.hAdd n 1) n) (CategoryTheory.CategoryStruct.comp (((AlgebraicT …
        -/
      · simp only [Fin.val_zero, pow_zero, one_zsmul]
        erw [Γ₀.Obj.mapMono_on_summand_id_assoc, Γ₀.Obj.Termwise.mapMono_δ₀,
          Splitting.cofan_inj_πSummand_eq_id, comp_id]
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.43, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K : ChainComplex C Nat
          n : Nat
          ⊢ ∀ (x : Fin (HAdd.hAdd n 2)), Ne x 0 → Eq (CategoryTheory.CategoryStruct.comp …
        -/
      · intro i hi
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.43, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K : ChainComplex C Nat
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          hi : Ne i 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
        -/
        dsimp
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.43, u_1} C
          inst✝¹ : CategoryTheory.Preadditive C
          inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
          K : ChainComplex C Nat
          n : Nat
          i : Fin (HAdd.hAdd n 2)
          hi : Ne i 0
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
        -/
        simp only [Preadditive.zsmul_comp, Preadditive.comp_zsmul, assoc]
        erw [Γ₀.Obj.mapMono_on_summand_id_assoc, Γ₀.Obj.Termwise.mapMono_eq_zero, zero_comp,
          zsmul_zero]
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.43, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            K : ChainComplex C Nat
            n : Nat
            i : Fin (HAdd.hAdd n 2)
            hi : Ne i 0
            ⊢ Ne (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
          -/
        · intro h
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.43, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            K : ChainComplex C Nat
            n : Nat
            i : Fin (HAdd.hAdd n 2)
            hi : Ne i 0
            h : Eq (SimplexCategory.mk (HAdd.hAdd n 1)) (SimplexCategory.mk n)
            ⊢ False
          -/
          replace h := congr_arg SimplexCategory.len h
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.43, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            K : ChainComplex C Nat
            n : Nat
            i : Fin (HAdd.hAdd n 2)
            hi : Ne i 0
            h : Eq (SimplexCategory.mk (HAdd.hAdd n 1)).len (SimplexCategory.mk n).len
            ⊢ False
          -/
          change n + 1 = n at h
          /-
            case h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.43, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            K : ChainComplex C Nat
            n : Nat
            i : Fin (HAdd.hAdd n 2)
            hi : Ne i 0
            h : Eq (HAdd.hAdd n 1) n
            ⊢ False
          -/
          omega
          /-
            🎉 no goals
          -/
          /-
            case h₂
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.43, u_1} C
            inst✝¹ : CategoryTheory.Preadditive C
            inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
            K : ChainComplex C Nat
            n : Nat
            i : Fin (HAdd.hAdd n 2)
            hi : Ne i 0
            ⊢ Not (AlgebraicTopology.DoldKan.Isδ₀ (SimplexCategory.δ i))
          -/
        · simpa only [Isδ₀.iff] using hi)
          /-
            🎉 no goals
          -/


/-- The natural isomorphism `(Γ₀.splitting K).nondegComplex ≅ K` for `K : ChainComplex C ℕ`. -/
def Γ₀'CompNondegComplexFunctor : Γ₀' ⋙ Split.nondegComplexFunctor ≅ 𝟭 (ChainComplex C ℕ) :=
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.7467, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ ∀ {X Y : ChainComplex C Nat} (f : Quiver.Hom X Y), Eq (CategoryTheory.Catego …
  -/
  NatIso.ofComponents Γ₀NondegComplexIso
  /-
    🎉 no goals
  -/


/-- The natural isomorphism `Γ₀ ⋙ N₁ ≅ toKaroubi (ChainComplex C ℕ)`. -/
def N₁Γ₀ : Γ₀ ⋙ N₁ ≅ toKaroubi (ChainComplex C ℕ) :=
  calc
    Γ₀ ⋙ N₁ ≅ Γ₀' ⋙ Split.forget C ⋙ N₁ := Functor.associator _ _ _
    _ ≅ Γ₀' ⋙ Split.nondegComplexFunctor ⋙ toKaroubi _ :=
      (isoWhiskerLeft Γ₀' Split.toKaroubiNondegComplexFunctorIsoN₁.symm)
    _ ≅ (Γ₀' ⋙ Split.nondegComplexFunctor) ⋙ toKaroubi _ := (Functor.associator _ _ _).symm
    _ ≅ 𝟭 _ ⋙ toKaroubi (ChainComplex C ℕ) := isoWhiskerRight Γ₀'CompNondegComplexFunctor _
    _ ≅ toKaroubi (ChainComplex C ℕ) := Functor.leftUnitor _


theorem N₁Γ₀_app (K : ChainComplex C ℕ) :
    N₁Γ₀.app K = (Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.symm ≪≫
      (toKaroubi _).mapIso (Γ₀NondegComplexIso K) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₁Γ₀.app K) ((AlgebraicTopology.DoldKan.Γ₀.spl …
  -/
  ext1
  /-
    case w
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₁Γ₀.app K).hom ((AlgebraicTopology.DoldKan.Γ₀ …
  -/
  dsimp [N₁Γ₀]
  /-
    case w
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [id_comp, comp_id, comp_id]
  /-
    case w
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (SimplicialObject.Split.toKaroubiNond …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem N₁Γ₀_hom_app (K : ChainComplex C ℕ) :
    N₁Γ₀.hom.app K = (Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.inv ≫
        (toKaroubi _).map (Γ₀NondegComplexIso K).hom := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₁Γ₀.hom.app K) (CategoryTheory.CategoryStruct …
  -/
  change (N₁Γ₀.app K).hom = _
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₁Γ₀.app K).hom (CategoryTheory.CategoryStruct …
  -/
  simp only [N₁Γ₀_app]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.s …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem N₁Γ₀_inv_app (K : ChainComplex C ℕ) :
    N₁Γ₀.inv.app K = (toKaroubi _).map (Γ₀NondegComplexIso K).inv ≫
        (Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.hom := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₁Γ₀.inv.app K) (CategoryTheory.CategoryStruct …
  -/
  change (N₁Γ₀.app K).inv = _
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₁Γ₀.app K).inv (CategoryTheory.CategoryStruct …
  -/
  simp only [N₁Γ₀_app]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.s …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem N₁Γ₀_hom_app_f_f (K : ChainComplex C ℕ) (n : ℕ) :
    (N₁Γ₀.hom.app K).f.f n = (Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.inv.f.f n := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.N₁Γ₀.hom.app K).f.f n) ((AlgebraicTopology.Do …
  -/
  rw [N₁Γ₀_hom_app]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Γ₀.splitt …
  -/
  apply comp_id
  /-
    🎉 no goals
  -/


@[simp]
theorem N₁Γ₀_inv_app_f_f (K : ChainComplex C ℕ) (n : ℕ) :
    (N₁Γ₀.inv.app K).f.f n = (Γ₀.splitting K).toKaroubiNondegComplexIsoN₁.hom.f.f n := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.N₁Γ₀.inv.app K).f.f n) ((AlgebraicTopology.Do …
  -/
  rw [N₁Γ₀_inv_app]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    K : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Idempotents.toKarou …
  -/
  apply id_comp
  /-
    🎉 no goals
  -/


/-- Compatibility isomorphism between `toKaroubi _ ⋙ Γ₂ ⋙ N₂` and `Γ₀ ⋙ N₁` which
are functors `ChainComplex C ℕ ⥤ Karoubi (ChainComplex C ℕ)`. -/
def N₂Γ₂ToKaroubiIso : toKaroubi (ChainComplex C ℕ) ⋙ Γ₂ ⋙ N₂ ≅ Γ₀ ⋙ N₁ :=
  calc
    toKaroubi (ChainComplex C ℕ) ⋙ Γ₂ ⋙ N₂ ≅
      toKaroubi (ChainComplex C ℕ) ⋙ (Γ₂ ⋙ N₂) := (Functor.associator _ _ _).symm
    _ ≅ (Γ₀ ⋙ toKaroubi (SimplicialObject C)) ⋙ N₂ :=
        isoWhiskerRight ((functorExtension₂CompWhiskeringLeftToKaroubiIso _ _).app Γ₀) N₂
    _ ≅ Γ₀ ⋙ toKaroubi (SimplicialObject C) ⋙ N₂ := Functor.associator _ _ _
    _ ≅ Γ₀ ⋙ N₁ :=
      isoWhiskerLeft Γ₀ ((functorExtension₁CompWhiskeringLeftToKaroubiIso _ _).app N₁)


@[simp]
lemma N₂Γ₂ToKaroubiIso_hom_app (X : ChainComplex C ℕ) :
    (N₂Γ₂ToKaroubiIso.hom.app X).f = PInfty := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₂Γ₂ToKaroubiIso.hom.app X).f AlgebraicTopolog …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.N₂Γ₂ToKaroubiIso.hom.app X).f.f n) (Algebraic …
  -/
  dsimp [N₂Γ₂ToKaroubiIso]
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [comp_id, assoc, PInfty_f_idem]
  conv_rhs =>
    rw [← PInfty_f_idem]
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  congr 1
  /-
    case h.e_a
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.Γ₀.splitt …
  -/
  apply (Γ₀.splitting X).hom_ext'
  /-
    case h.e_a.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }) …
  -/
  intro A
  /-
    case h.e_a.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  rw [Splitting.ι_desc_assoc, assoc]
  /-
    case h.e_a.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (X. …
  -/
  apply id_comp
  /-
    🎉 no goals
  -/


@[simp]
lemma N₂Γ₂ToKaroubiIso_inv_app (X : ChainComplex C ℕ) :
    (N₂Γ₂ToKaroubiIso.inv.app X).f = PInfty := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.N₂Γ₂ToKaroubiIso.inv.app X).f AlgebraicTopolog …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.N₂Γ₂ToKaroubiIso.inv.app X).f.f n) (Algebraic …
  -/
  dsimp [N₂Γ₂ToKaroubiIso]
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  simp only [comp_id, PInfty_f_idem_assoc, AlternatingFaceMapComplex.obj_X, Γ₀_obj_obj]
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  convert comp_id _
  /-
    case h.e'_2.h.e'_7
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Γ₀.splitting X).desc { unop := SimplexCategor …
  -/
  apply (Γ₀.splitting X).hom_ext'
  /-
    case h.e'_2.h.e'_7.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    ⊢ ∀ (A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }) …
  -/
  intro A
  /-
    case h.e'_2.h.e'_7.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.DoldKan.Γ₀.split …
  -/
  rw [Splitting.ι_desc]
  /-
    case h.e'_2.h.e'_7.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : ChainComplex C Nat
    n : Nat
    A : SimplicialObject.Splitting.IndexSet { unop := SimplexCategory.mk n }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (X. …
  -/
  erw [comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- The counit isomorphism of the Dold-Kan equivalence for additive categories. -/
def N₂Γ₂ : Γ₂ ⋙ N₂ ≅ 𝟭 (Karoubi (ChainComplex C ℕ)) :=
  ((whiskeringLeft _ _ _).obj (toKaroubi (ChainComplex C ℕ))).preimageIso
      (N₂Γ₂ToKaroubiIso ≪≫ N₁Γ₀)


@[simp]
theorem N₂Γ₂_inv_app_f_f (X : Karoubi (ChainComplex C ℕ)) (n : ℕ) :
    (N₂Γ₂.inv.app X).f.f n =
      X.p.f n ≫ ((Γ₀.splitting X.X).cofan _).inj (Splitting.IndexSet.id (op [n])) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.Idempotents.Karoubi (ChainComplex C Nat)
    n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.N₂Γ₂.inv.app X).f.f n) (CategoryTheory.Catego …
  -/
  dsimp [N₂Γ₂]
  simp only [whiskeringLeft_obj_preimage_app, NatTrans.comp_app, Functor.comp_map,
    Karoubi.comp_f, N₂Γ₂ToKaroubiIso_inv_app, HomologicalComplex.comp_f,
    N₁Γ₀_inv_app_f_f, toKaroubi_obj_X, Splitting.toKaroubiNondegComplexIsoN₁_hom_f_f,
    Γ₀.obj_obj, PInfty_on_Γ₀_splitting_summand_eq_self, N₂_map_f_f,
    Γ₂_map_f_app, unop_op, Karoubi.decompId_p_f, PInfty_f_idem_assoc,
    PInfty_on_Γ₀_splitting_summand_eq_self_assoc, Splitting.IndexSet.id_fst, SimplexCategory.len_mk,
    Splitting.ι_desc]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    X : CategoryTheory.Idempotents.Karoubi (ChainComplex C Nat)
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.id (Categor …
  -/
  apply Karoubi.HomologicalComplex.p_idem_assoc
  /-
    🎉 no goals
  -/

-- Porting note: added to ease the proof of `N₂Γ₂_compatible_with_N₁Γ₀`

lemma whiskerLeft_toKaroubi_N₂Γ₂_hom :
    whiskerLeft (toKaroubi (ChainComplex C ℕ)) N₂Γ₂.hom = N₂Γ₂ToKaroubiIso.hom ≫ N₁Γ₀.hom := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    ⊢ Eq (CategoryTheory.whiskerLeft (CategoryTheory.Idempotents.toKaroubi (ChainC …
  -/
  let e : _ ≅ toKaroubi (ChainComplex C ℕ) ⋙ 𝟭 _ := N₂Γ₂ToKaroubiIso ≪≫ N₁Γ₀
  have h := ((whiskeringLeft _ _ (Karoubi (ChainComplex C ℕ))).obj
    (toKaroubi (ChainComplex C ℕ))).map_preimage e.hom
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    e : CategoryTheory.Iso ((CategoryTheory.Idempotents.toKaroubi (ChainComplex C  …
    h : Eq (((CategoryTheory.whiskeringLeft (ChainComplex C Nat) (CategoryTheory.I …
    ⊢ Eq (CategoryTheory.whiskerLeft (CategoryTheory.Idempotents.toKaroubi (ChainC …
  -/
  dsimp only [whiskeringLeft, N₂Γ₂, Functor.preimageIso] at h ⊢
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
    e : CategoryTheory.Iso ((CategoryTheory.Idempotents.toKaroubi (ChainComplex C  …
    h : Eq (CategoryTheory.whiskerLeft (CategoryTheory.Idempotents.toKaroubi (Chai …
    ⊢ Eq (CategoryTheory.whiskerLeft (CategoryTheory.Idempotents.toKaroubi (ChainC …
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem N₂Γ₂_compatible_with_N₁Γ₀ (K : ChainComplex C ℕ) :
    N₂Γ₂.hom.app ((toKaroubi _).obj K) = N₂Γ₂ToKaroubiIso.hom.app K ≫ N₁Γ₀.hom.app K :=
  congr_app whiskerLeft_toKaroubi_N₂Γ₂_hom K


