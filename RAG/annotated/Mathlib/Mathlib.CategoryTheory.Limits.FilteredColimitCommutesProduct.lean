/-- Given a family of functors `I i ⥤ C` for `i : α`, we obtain a functor `(∀ i, I i) ⥤ C` which
maps `k : ∀ i, I i` to `∏ᶜ fun (s : α) => (F s).obj (k s)`. -/
@[simps]
noncomputable def pointwiseProduct : (∀ i, I i) ⥤ C where
  obj k := ∏ᶜ fun (s : α) => (F s).obj (k s)
  map f := Pi.map (fun s => (F s).map (f s))


/-- The inclusions `(F s).obj (k s) ⟶ colimit (F s)` induce a cocone on `pointwiseProduct F` with
cone point `∏ᶜ (fun s : α) => colimit (F s)`. -/
@[simps]
noncomputable def coconePointwiseProduct : Cocone (pointwiseProduct F) where
  pt := ∏ᶜ fun (s : α) => colimit (F s)
  ι := { app := fun k => Pi.map fun s => colimit.ι _ _ }


/-- The natural morphism `colim_k (∏ᶜ s ↦ (F s).obj (k s)) ⟶ ∏ᶜ s ↦ colim_k (F s).obj (k s)`.
We will say that a category has the `IPC` property if this morphism is an isomorphism as long
as the indexing categories are filtered. -/
noncomputable def colimitPointwiseProductToProductColimit :
    colimit (pointwiseProduct F) ⟶ ∏ᶜ fun (s : α) => colimit (F s) :=
  colimit.desc (pointwiseProduct F) (coconePointwiseProduct F)


@[reassoc (attr := simp)]
theorem ι_colimitPointwiseProductToProductColimit_π (k : ∀ i, I i) (s : α) :
    colimit.ι (pointwiseProduct F) k ≫ colimitPointwiseProductToProductColimit F ≫ Pi.π _ s =
      Pi.π _ s ≫ colimit.ι (F s) (k s) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    α : Type w
    I : α → Type u₁
    inst✝³ : (i : α) → CategoryTheory.Category.{v₁, u₁} (I i)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    F : (i : α) → CategoryTheory.Functor (I i) C
    inst✝¹ : ∀ (i : α), CategoryTheory.Limits.HasColimitsOfShape (I i) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape ((i : α) → I i) C
    k : (i : α) → I i
    s : α
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  simp [colimitPointwiseProductToProductColimit]
  /-
    🎉 no goals
  -/


/-- Evaluating the pointwise product `k ↦ ∏ᶜ fun (s : α) => (F s).obj (k s)` at `d` is the same as
taking the pointwise product `k ↦ ∏ᶜ fun (s : α) => ((F s).obj (k s)).obj d`. -/
@[simps!]
noncomputable def pointwiseProductCompEvaluation (d : D) :
    pointwiseProduct F ⋙ (evaluation D C).obj d ≅
      pointwiseProduct (fun s => F s ⋙ (evaluation _ _).obj d) :=
  NatIso.ofComponents (fun k => piObjIso _ _)
                                 /-
                                   C : Type u
                                   inst✝³ : CategoryTheory.Category.{v, u} C
                                   D : Type u₁
                                   inst✝² : CategoryTheory.Category.{v₁, u₁} D
                                   α : Type w
                                   I : α → Type u₂
                                   inst✝¹ : (i : α) → CategoryTheory.Category.{w, u₂} (I i)
                                   inst✝ : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
                                   F : (i : α) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
                                   d : D
                                   X✝ Y✝ : (i : α) → I i
                                   f : Quiver.Hom X✝ Y✝
                                   ⊢ ∀ (b : α), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategorySt …
                                 -/
    (fun f => Pi.hom_ext _ _ (by simp [← NatTrans.comp_app]))
                                 /-
                                   🎉 no goals
                                 -/


theorem colimitPointwiseProductToProductColimit_app (d : D) :
    (colimitPointwiseProductToProductColimit F).app d =
      (colimitObjIsoColimitCompEvaluation _ _).hom ≫
        (HasColimit.isoOfNatIso (pointwiseProductCompEvaluation F d)).hom ≫
          colimitPointwiseProductToProductColimit _ ≫
            (Pi.mapIso fun _ => (colimitObjIsoColimitCompEvaluation _ _).symm).hom ≫
              (piObjIso _ _).inv := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    I : α → Type u₂
    inst✝³ : (i : α) → CategoryTheory.Category.{w, u₂} (I i)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    F : (i : α) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
    inst✝¹ : ∀ (i : α), CategoryTheory.Limits.HasColimitsOfShape (I i) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape ((i : α) → I i) C
    d : D
    ⊢ Eq ((CategoryTheory.Limits.colimitPointwiseProductToProductColimit F).app d) …
  -/
  rw [← Iso.inv_comp_eq]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    I : α → Type u₂
    inst✝³ : (i : α) → CategoryTheory.Category.{w, u₂} (I i)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    F : (i : α) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
    inst✝¹ : ∀ (i : α), CategoryTheory.Limits.HasColimitsOfShape (I i) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape ((i : α) → I i) C
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitObjIsoC …
  -/
  simp only [← Category.assoc]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    I : α → Type u₂
    inst✝³ : (i : α) → CategoryTheory.Category.{w, u₂} (I i)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    F : (i : α) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
    inst✝¹ : ∀ (i : α), CategoryTheory.Limits.HasColimitsOfShape (I i) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape ((i : α) → I i) C
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimitObjIsoC …
  -/
  rw [Iso.eq_comp_inv]
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    I : α → Type u₂
    inst✝³ : (i : α) → CategoryTheory.Category.{w, u₂} (I i)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    F : (i : α) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
    inst✝¹ : ∀ (i : α), CategoryTheory.Limits.HasColimitsOfShape (I i) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape ((i : α) → I i) C
    d : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  refine Pi.hom_ext _ _ (fun s => colimit.hom_ext (fun k => ?_))
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} D
    α : Type w
    I : α → Type u₂
    inst✝³ : (i : α) → CategoryTheory.Category.{w, u₂} (I i)
    inst✝² : CategoryTheory.Limits.HasLimitsOfShape (CategoryTheory.Discrete α) C
    F : (i : α) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
    inst✝¹ : ∀ (i : α), CategoryTheory.Limits.HasColimitsOfShape (I i) C
    inst✝ : CategoryTheory.Limits.HasColimitsOfShape ((i : α) → I i) C
    d : D
    s : α
    k : (i : α) → I i
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((Ca …
  -/
  simp [← NatTrans.comp_app]
  /-
    🎉 no goals
  -/


/-- A category `C` has the `w`-IPC property if the natural morphism
`colim_k (∏ᶜ s ↦ (F s).obj (k s)) ⟶ ∏ᶜ s ↦ colim_k (F s).obj (k s)` is an isomorphism for any
family of functors `F i : I i ⥤ C` with `I i` `w`-small and filtered for all `i`. -/
class IsIPC [HasProducts.{w} C] [HasFilteredColimitsOfSize.{w} C] : Prop where
  /-- `colimitPointwiseProductToProductColimit F` is always an isomorphism. -/
  isIso : ∀ (α : Type w) (I : α → Type w) [∀ i, SmallCategory (I i)] [∀ i, IsFiltered (I i)]
    (F : ∀ i, I i ⥤ C), IsIso (colimitPointwiseProductToProductColimit F)


theorem Types.isIso_colimitPointwiseProductToProductColimit (F : ∀ i, I i ⥤ Type u) :
    IsIso (colimitPointwiseProductToProductColimit F) := by
  -- We follow the proof in [Kashiwara2006], Prop. 3.1.11(ii)
  /-
    α : Type u
    I : α → Type u
    inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
    inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
    F : (i : α) → CategoryTheory.Functor (I i) (Type u)
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.colimitPointwiseProductToProduct …
  -/
  refine (isIso_iff_bijective _).2 ⟨fun y y' hy => ?_, fun x => ?_⟩
    /-
      case refine_1
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      y y' : CategoryTheory.Limits.colimit (CategoryTheory.Limits.pointwiseProduct F)
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F y) (C …
      ⊢ Eq y y'
    -/
  · obtain ⟨ky, yk₀, hyk₀⟩ := Types.jointly_surjective' y
    /-
      case refine_1.intro.intro
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      y y' : CategoryTheory.Limits.colimit (CategoryTheory.Limits.pointwiseProduct F)
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F y) (C …
      ky : (i : α) → I i
      yk₀ : (CategoryTheory.Limits.pointwiseProduct F).obj ky
      hyk₀ : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePro …
      ⊢ Eq y y'
    -/
    obtain ⟨ky', yk₀', hyk₀'⟩ := Types.jointly_surjective' y'
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      y y' : CategoryTheory.Limits.colimit (CategoryTheory.Limits.pointwiseProduct F)
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F y) (C …
      ky : (i : α) → I i
      yk₀ : (CategoryTheory.Limits.pointwiseProduct F).obj ky
      hyk₀ : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePro …
      ky' : (i : α) → I i
      yk₀' : (CategoryTheory.Limits.pointwiseProduct F).obj ky'
      hyk₀' : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePr …
      ⊢ Eq y y'
    -/
    let k := IsFiltered.max ky ky'
    let yk : (pointwiseProduct F).obj k :=
      (pointwiseProduct F).map (IsFiltered.leftToMax ky ky') yk₀
    let yk' : (pointwiseProduct F).obj k :=
      (pointwiseProduct F).map (IsFiltered.rightToMax ky ky') yk₀'
    obtain rfl : y = colimit.ι (pointwiseProduct F) k yk := by
      simp only [k, yk, Types.Colimit.w_apply', hyk₀]
    obtain rfl : y' = colimit.ι (pointwiseProduct F) k yk' := by
      simp only [k, yk', Types.Colimit.w_apply', hyk₀']
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      ky : (i : α) → I i
      yk₀ : (CategoryTheory.Limits.pointwiseProduct F).obj ky
      ky' : (i : α) → I i
      yk₀' : (CategoryTheory.Limits.pointwiseProduct F).obj ky'
      k : (i : α) → I i := CategoryTheory.IsFiltered.max ky ky'
      yk : (CategoryTheory.Limits.pointwiseProduct F).obj k := (CategoryTheory.Limit …
      yk' : (CategoryTheory.Limits.pointwiseProduct F).obj k := (CategoryTheory.Limi …
      hyk₀ : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePro …
      hyk₀' : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePr …
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F (Cate …
      ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwiseProduct  …
    -/
    dsimp only [pointwiseProduct_obj] at yk yk'
    have hch : ∀ (s : α), ∃ (i' : I s) (hi' : k s ⟶ i'),
        (F s).map hi' (Pi.π (fun s => (F s).obj (k s)) s yk) =
          (F s).map hi' (Pi.π (fun s => (F s).obj (k s)) s yk') := by
      intro s
      have hy₁ := congrFun (ι_colimitPointwiseProductToProductColimit_π F k s) yk
      have hy₂ := congrFun (ι_colimitPointwiseProductToProductColimit_π F k s) yk'
      dsimp only [pointwiseProduct_obj, types_comp_apply] at hy₁ hy₂
      rw [← hy, hy₁, Types.FilteredColimit.colimit_eq_iff] at hy₂
      obtain ⟨i₀, f₀, g₀, h₀⟩ := hy₂
      refine ⟨IsFiltered.coeq f₀ g₀, f₀ ≫ IsFiltered.coeqHom f₀ g₀, ?_⟩
      conv_rhs => rw [IsFiltered.coeq_condition]
      simp [h₀]
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      ky : (i : α) → I i
      yk₀ : (CategoryTheory.Limits.pointwiseProduct F).obj ky
      ky' : (i : α) → I i
      yk₀' : (CategoryTheory.Limits.pointwiseProduct F).obj ky'
      k : (i : α) → I i := CategoryTheory.IsFiltered.max ky ky'
      yk : CategoryTheory.Limits.piObj fun s => (F s).obj (k s) := (CategoryTheory.L …
      yk' : CategoryTheory.Limits.piObj fun s => (F s).obj (k s) := (CategoryTheory. …
      hyk₀ : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePro …
      hyk₀' : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePr …
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F (Cate …
      hch : ∀ (s : α), Exists fun i' => Exists fun hi' => Eq ((F s).map hi' (Categor …
      ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwiseProduct  …
    -/
    choose k' f hk' using hch
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      ky : (i : α) → I i
      yk₀ : (CategoryTheory.Limits.pointwiseProduct F).obj ky
      ky' : (i : α) → I i
      yk₀' : (CategoryTheory.Limits.pointwiseProduct F).obj ky'
      k : (i : α) → I i := CategoryTheory.IsFiltered.max ky ky'
      yk : CategoryTheory.Limits.piObj fun s => (F s).obj (k s) := (CategoryTheory.L …
      yk' : CategoryTheory.Limits.piObj fun s => (F s).obj (k s) := (CategoryTheory. …
      hyk₀ : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePro …
      hyk₀' : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePr …
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F (Cate …
      k' : (s : α) → I s
      f : (s : α) → Quiver.Hom (k s) (k' s)
      hk' : ∀ (s : α), Eq ((F s).map (f s) (CategoryTheory.Limits.Pi.π (fun s => (F  …
      ⊢ Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwiseProduct  …
    -/
    apply Types.colimit_sound' f f
    /-
      case refine_1.intro.intro.intro.intro
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      ky : (i : α) → I i
      yk₀ : (CategoryTheory.Limits.pointwiseProduct F).obj ky
      ky' : (i : α) → I i
      yk₀' : (CategoryTheory.Limits.pointwiseProduct F).obj ky'
      k : (i : α) → I i := CategoryTheory.IsFiltered.max ky ky'
      yk : CategoryTheory.Limits.piObj fun s => (F s).obj (k s) := (CategoryTheory.L …
      yk' : CategoryTheory.Limits.piObj fun s => (F s).obj (k s) := (CategoryTheory. …
      hyk₀ : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePro …
      hyk₀' : Eq (CategoryTheory.Limits.colimit.ι (CategoryTheory.Limits.pointwisePr …
      hy : Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F (Cate …
      k' : (s : α) → I s
      f : (s : α) → Quiver.Hom (k s) (k' s)
      hk' : ∀ (s : α), Eq ((F s).map (f s) (CategoryTheory.Limits.Pi.π (fun s => (F  …
      ⊢ Eq ((CategoryTheory.Limits.pointwiseProduct F).map f yk) ((CategoryTheory.Li …
    -/
    exact Types.limit_ext' _ _ _ (fun ⟨s⟩ => by simpa using hk' _)
    /-
      🎉 no goals
    -/
  · have hch : ∀ (s : α), ∃ (i : I s) (xi : (F s).obj i), colimit.ι (F s) i xi =
        Pi.π (fun s => colimit (F s)) s x := fun s => Types.jointly_surjective' _
    /-
      case refine_2
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      x : CategoryTheory.Limits.piObj fun s => CategoryTheory.Limits.colimit (F s)
      hch : ∀ (s : α), Exists fun i => Exists fun xi => Eq (CategoryTheory.Limits.co …
      ⊢ Exists fun a => Eq (CategoryTheory.Limits.colimitPointwiseProductToProductCo …
    -/
    choose k p hk using hch
    /-
      case refine_2
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      x : CategoryTheory.Limits.piObj fun s => CategoryTheory.Limits.colimit (F s)
      k : (s : α) → I s
      p : (s : α) → (F s).obj (k s)
      hk : ∀ (s : α), Eq (CategoryTheory.Limits.colimit.ι (F s) (k s) (p s)) (Catego …
      ⊢ Exists fun a => Eq (CategoryTheory.Limits.colimitPointwiseProductToProductCo …
    -/
    refine ⟨colimit.ι (pointwiseProduct F) k ((Types.productIso _).inv p), ?_⟩
    /-
      case refine_2
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      x : CategoryTheory.Limits.piObj fun s => CategoryTheory.Limits.colimit (F s)
      k : (s : α) → I s
      p : (s : α) → (F s).obj (k s)
      hk : ∀ (s : α), Eq (CategoryTheory.Limits.colimit.ι (F s) (k s) (p s)) (Catego …
      ⊢ Eq (CategoryTheory.Limits.colimitPointwiseProductToProductColimit F (Categor …
    -/
    refine Types.limit_ext' _ _ _ (fun ⟨s⟩ => ?_)
    have := congrFun (ι_colimitPointwiseProductToProductColimit_π F k s)
      ((Types.productIso _).inv p)
    /-
      case refine_2
      α : Type u
      I : α → Type u
      inst✝¹ : (i : α) → CategoryTheory.SmallCategory (I i)
      inst✝ : ∀ (i : α), CategoryTheory.IsFiltered (I i)
      F : (i : α) → CategoryTheory.Functor (I i) (Type u)
      x : CategoryTheory.Limits.piObj fun s => CategoryTheory.Limits.colimit (F s)
      k : (s : α) → I s
      p : (s : α) → (F s).obj (k s)
      hk : ∀ (s : α), Eq (CategoryTheory.Limits.colimit.ι (F s) (k s) (p s)) (Catego …
      x✝ : CategoryTheory.Discrete α
      s : α
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι …
      ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Discrete.functor fun s =>  …
    -/
    exact this.trans (by simpa using hk _)
    /-
      🎉 no goals
    -/


instance : IsIPC.{u} (Type u) where
  isIso _ _ := Types.isIso_colimitPointwiseProductToProductColimit


instance [HasProducts.{w} C] [HasFilteredColimitsOfSize.{w, w} C] [IsIPC.{w} C] {D : Type u₁}
    [Category.{v₁} D] : IsIPC.{w} (D ⥤ C) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasProducts C
    inst✝² : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v, u} C
    inst✝¹ : CategoryTheory.Limits.IsIPC C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    ⊢ CategoryTheory.Limits.IsIPC (CategoryTheory.Functor D C)
  -/
  refine ⟨fun β I _ _ F => ?_⟩
  suffices ∀ d, IsIso ((colimitPointwiseProductToProductColimit F).app d) from
    NatIso.isIso_of_isIso_app _
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasProducts C
    inst✝² : CategoryTheory.Limits.HasFilteredColimitsOfSize.{w, w, v, u} C
    inst✝¹ : CategoryTheory.Limits.IsIPC C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    β : Type w
    I : β → Type w
    x✝¹ : (i : β) → CategoryTheory.SmallCategory (I i)
    x✝ : ∀ (i : β), CategoryTheory.IsFiltered (I i)
    F : (i : β) → CategoryTheory.Functor (I i) (CategoryTheory.Functor D C)
    ⊢ ∀ (d : D), CategoryTheory.IsIso ((CategoryTheory.Limits.colimitPointwiseProd …
  -/
  exact fun d => colimitPointwiseProductToProductColimit_app F d ▸ inferInstance
  /-
    🎉 no goals
  -/


