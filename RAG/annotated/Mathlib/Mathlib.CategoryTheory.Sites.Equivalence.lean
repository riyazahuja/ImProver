instance (priority := 900) [G.IsEquivalence] : IsCoverDense G J where
  is_cover U := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      ⊢ Membership.mem (J U) (CategoryTheory.Sieve.coverByImage G U)
    -/
    let e := (asEquivalence G).symm
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e✝ : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      e : CategoryTheory.Equivalence C D := G.asEquivalence.symm
      ⊢ Membership.mem (J U) (CategoryTheory.Sieve.coverByImage G U)
    -/
    convert J.top_mem U
    /-
      case h.e'_5
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e✝ : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      e : CategoryTheory.Equivalence C D := G.asEquivalence.symm
      ⊢ Eq (CategoryTheory.Sieve.coverByImage G U) Top.top
    -/
    ext Y f
    simp only [Sieve.functorPushforward_apply, Presieve.functorPushforward, exists_and_left,
      Sieve.top_apply, iff_true]
    /-
      case h.e'_5.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e✝ : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      e : CategoryTheory.Equivalence C D := G.asEquivalence.symm
      Y : C
      f : Quiver.Hom Y U
      ⊢ (CategoryTheory.Sieve.coverByImage G U).arrows f
    -/
    let g : e.inverse.obj _ ⟶ U := (e.unitInv.app Y) ≫ f
    /-
      case h.e'_5.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e✝ : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      e : CategoryTheory.Equivalence C D := G.asEquivalence.symm
      Y : C
      f : Quiver.Hom Y U
      g : Quiver.Hom (e.inverse.obj (e.functor.obj Y)) U := CategoryTheory.CategoryS …
      ⊢ (CategoryTheory.Sieve.coverByImage G U).arrows f
    -/
    have : (Sieve.coverByImage e.inverse U).arrows g := Presieve.in_coverByImage _ g
    /-
      case h.e'_5.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e✝ : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      e : CategoryTheory.Equivalence C D := G.asEquivalence.symm
      Y : C
      f : Quiver.Hom Y U
      g : Quiver.Hom (e.inverse.obj (e.functor.obj Y)) U := CategoryTheory.CategoryS …
      this : (CategoryTheory.Sieve.coverByImage e.inverse U).arrows g
      ⊢ (CategoryTheory.Sieve.coverByImage G U).arrows f
    -/
    replace := Sieve.downward_closed _ this (e.unit.app Y)
    /-
      case h.e'_5.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e✝ : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : G.IsEquivalence
      U : C
      e : CategoryTheory.Equivalence C D := G.asEquivalence.symm
      Y : C
      f : Quiver.Hom Y U
      g : Quiver.Hom (e.inverse.obj (e.functor.obj Y)) U := CategoryTheory.CategoryS …
      this : (CategoryTheory.Sieve.coverByImage e.inverse U).arrows (CategoryTheory. …
      ⊢ (CategoryTheory.Sieve.coverByImage G U).arrows f
    -/
    simpa [g] using this
    /-
      🎉 no goals
    -/


instance : e.functor.IsDenseSubsite J (e.inverse.inducedTopology J) := by
  have : J = e.functor.inducedTopology (e.inverse.inducedTopology J) := by
    ext X S
    rw [show S ∈ (e.functor.inducedTopology (e.inverse.inducedTopology J)) X ↔ _
      from (GrothendieckTopology.pullback_mem_iff_of_isIso (i := e.unit.app X)).symm]
    congr!; ext Y f; simp
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    e : CategoryTheory.Equivalence C D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} A
    this : Eq J (e.functor.inducedTopology (e.inverse.inducedTopology J))
    ⊢ CategoryTheory.Functor.IsDenseSubsite J (e.inverse.inducedTopology J) e.func …
  -/
  nth_rw 1 [this]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    e : CategoryTheory.Equivalence C D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} A
    this : Eq J (e.functor.inducedTopology (e.inverse.inducedTopology J))
    ⊢ CategoryTheory.Functor.IsDenseSubsite (e.functor.inducedTopology (e.inverse. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma eq_inducedTopology_of_isDenseSubsite [e.inverse.IsDenseSubsite K J] :
    K = e.inverse.inducedTopology J := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    e : CategoryTheory.Equivalence C D
    inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
    ⊢ Eq K (e.inverse.inducedTopology J)
  -/
  ext
  /-
    case h.h.h
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    e : CategoryTheory.Equivalence C D
    inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
    x✝¹ : D
    x✝ : CategoryTheory.Sieve x✝¹
    ⊢ Iff (Membership.mem (K x✝¹) x✝) (Membership.mem ((e.inverse.inducedTopology  …
  -/
  exact (e.inverse.functorPushforward_mem_iff K J).symm
  /-
    🎉 no goals
  -/


instance : e.functor.IsDenseSubsite J K := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    e : CategoryTheory.Equivalence C D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
    inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
    ⊢ CategoryTheory.Functor.IsDenseSubsite J K e.functor
  -/
  rw [e.eq_inducedTopology_of_isDenseSubsite J K]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    e : CategoryTheory.Equivalence C D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
    inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
    ⊢ CategoryTheory.Functor.IsDenseSubsite J (e.inverse.inducedTopology J) e.func …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The functor in the equivalence of sheaf categories. -/
@[simps!]
def sheafCongr.functor : Sheaf J A ⥤ Sheaf K A where
  obj F := ⟨e.inverse.op ⋙ F.val, e.inverse.op_comp_isSheaf _ _ _⟩
  map f := ⟨whiskerLeft e.inverse.op f.val⟩


/-- The inverse in the equivalence of sheaf categories. -/
@[simps!]
def sheafCongr.inverse : Sheaf K A ⥤ Sheaf J A where
  obj F := ⟨e.functor.op ⋙ F.val, e.functor.op_comp_isSheaf _ _ _⟩
  map f := ⟨whiskerLeft e.functor.op f.val⟩


/-- The unit iso in the equivalence of sheaf categories. -/
@[simps!]
def sheafCongr.unitIso : 𝟭 (Sheaf J A) ≅ functor J K e A ⋙ inverse J K e A :=
  NatIso.ofComponents (fun F ↦ ⟨⟨(isoWhiskerRight e.op.unitIso F.val).hom⟩,
    ⟨(isoWhiskerRight e.op.unitIso F.val).inv⟩,
    Sheaf.hom_ext _ _ (isoWhiskerRight e.op.unitIso F.val).hom_inv_id,
                                                                             /-
                                                                               C : Type u₁
                                                                               inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                               J : CategoryTheory.GrothendieckTopology C
                                                                               D : Type u₂
                                                                               inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                               K : CategoryTheory.GrothendieckTopology D
                                                                               e : CategoryTheory.Equivalence C D
                                                                               G : CategoryTheory.Functor D C
                                                                               A : Type u₃
                                                                               inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
                                                                               inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
                                                                               ⊢ ∀ {X Y : CategoryTheory.Sheaf J A} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
                                                                             -/
    Sheaf.hom_ext _ _ (isoWhiskerRight e.op.unitIso F.val).inv_hom_id⟩ ) (by aesop)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- The counit iso in the equivalence of sheaf categories. -/
@[simps!]
def sheafCongr.counitIso : inverse J K e A ⋙ functor J K e A ≅ 𝟭 (Sheaf _ A) :=
  NatIso.ofComponents (fun F ↦ ⟨⟨(isoWhiskerRight e.op.counitIso F.val).hom⟩,
    ⟨(isoWhiskerRight e.op.counitIso F.val).inv⟩,
    Sheaf.hom_ext _ _ (isoWhiskerRight e.op.counitIso F.val).hom_inv_id,
                                                                               /-
                                                                                 C : Type u₁
                                                                                 inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                                 J : CategoryTheory.GrothendieckTopology C
                                                                                 D : Type u₂
                                                                                 inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                                 K : CategoryTheory.GrothendieckTopology D
                                                                                 e : CategoryTheory.Equivalence C D
                                                                                 G : CategoryTheory.Functor D C
                                                                                 A : Type u₃
                                                                                 inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
                                                                                 inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
                                                                                 ⊢ ∀ {X Y : CategoryTheory.Sheaf K A} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
                                                                               -/
    Sheaf.hom_ext _ _ (isoWhiskerRight e.op.counitIso F.val).inv_hom_id⟩ ) (by aesop)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The equivalence of sheaf categories. -/
def sheafCongr : Sheaf J A ≌ Sheaf K A where
  functor := sheafCongr.functor J K e A
  inverse := sheafCongr.inverse J K e A
  unitIso := sheafCongr.unitIso J K e A
  counitIso := sheafCongr.counitIso J K e A
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
      X : CategoryTheory.Sheaf J A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Equivalence.sheafCon …
    -/
    ext
    simp only [id_obj, sheafCongr.functor_obj_val_obj, comp_obj,
      Sheaf.instCategorySheaf_comp_val, NatTrans.comp_app, sheafCongr.inverse_obj_val_obj,
      Opposite.unop_op, sheafCongr.functor_map_val_app,
      sheafCongr.unitIso_hom_app_val_app, sheafCongr.counitIso_hom_app_val_app,
      sheafCongr.functor_obj_val_map, Quiver.Hom.unop_op, Sheaf.instCategorySheaf_id_val,
      NatTrans.id_app]
    /-
      case h.w.h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝¹ : CategoryTheory.Category.{v₃, u₃} A
      inst✝ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
      X : CategoryTheory.Sheaf J A
      x✝ : Opposite D
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.val.map (e.unitIso.inv.app (e.inve …
    -/
    simp [← Functor.map_comp, ← op_comp]
    /-
      🎉 no goals
    -/


/-- Transport a presheaf to the equivalent category and sheafify there. -/
noncomputable
def transportAndSheafify : (Cᵒᵖ ⥤ A) ⥤ Sheaf J A :=
  e.op.congrLeft.functor ⋙ presheafToSheaf _ _ ⋙ (e.sheafCongr J K A).inverse


/-- An auxiliary definition for the sheafification adjunction. -/
noncomputable
def transportIsoSheafToPresheaf : (e.sheafCongr J K A).functor ⋙
    sheafToPresheaf K A ⋙ e.op.congrLeft.inverse ≅ sheafToPresheaf J A :=
  NatIso.ofComponents (fun F ↦ isoWhiskerRight e.op.unitIso.symm F.val)
        /-
          C : Type u₁
          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
          J : CategoryTheory.GrothendieckTopology C
          D : Type u₂
          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
          K : CategoryTheory.GrothendieckTopology D
          e : CategoryTheory.Equivalence C D
          G : CategoryTheory.Functor D C
          A : Type u₃
          inst✝² : CategoryTheory.Category.{v₃, u₃} A
          inst✝¹ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
          inst✝ : CategoryTheory.HasSheafify K A
          ⊢ ∀ {X Y : CategoryTheory.Sheaf J A} (f : Quiver.Hom X Y), Eq (CategoryTheory. …
        -/
    (by intros; ext; simp [Equivalence.sheafCongr])
                     /-
                       🎉 no goals
                     -/


/-- Transporting and sheafifying is left adjoint to taking the underlying presheaf. -/
noncomputable
def transportSheafificationAdjunction : transportAndSheafify J K e A ⊣ sheafToPresheaf J A :=
  ((e.op.congrLeft.toAdjunction.comp (sheafificationAdjunction _ _)).comp
    (e.sheafCongr J K A).symm.toAdjunction).ofNatIsoRight
    (transportIsoSheafToPresheaf _ _ _ _)


noncomputable instance : PreservesFiniteLimits <| transportAndSheafify J K e A where
  preservesFiniteLimits _ := comp_preservesLimitsOfShape _ _


include K e in
/-- Transport `HasSheafify` along an equivalence of sites. -/
theorem hasSheafify : HasSheafify J A :=
  HasSheafify.mk' J A (transportSheafificationAdjunction J K e A)


include K e in
theorem hasSheafCompose : J.HasSheafCompose F where
  isSheaf P hP := by
    have hP' : Presheaf.IsSheaf K (e.inverse.op ⋙ P ⋙ F) := by
      change Presheaf.IsSheaf K ((_ ⋙ _) ⋙ _)
      apply HasSheafCompose.isSheaf
      exact e.inverse.op_comp_isSheaf K J ⟨P, hP⟩
    replace hP' : Presheaf.IsSheaf J (e.functor.op ⋙ e.inverse.op ⋙ P ⋙ F) :=
      e.functor.op_comp_isSheaf _ _ ⟨_, hP'⟩
    /-
      C : Type u₁
      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      e : CategoryTheory.Equivalence C D
      inst✝³ : CategoryTheory.Functor.IsDenseSubsite K J e.inverse
      A : Type u_1
      inst✝² : CategoryTheory.Category.{u_3, u_1} A
      B : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_4, u_2} B
      F : CategoryTheory.Functor A B
      inst✝ : K.HasSheafCompose F
      P : CategoryTheory.Functor (Opposite C) A
      hP : CategoryTheory.Presheaf.IsSheaf J P
      hP' : CategoryTheory.Presheaf.IsSheaf J (e.functor.op.comp (e.inverse.op.comp  …
      ⊢ CategoryTheory.Presheaf.IsSheaf J (P.comp F)
    -/
    exact (Presheaf.isSheaf_of_iso_iff ((isoWhiskerRight e.op.unitIso.symm (P ⋙ F)))).mp hP'
    /-
      🎉 no goals
    -/


/-- Transport to a small model and sheafify there. -/
noncomputable
def smallSheafify : (Cᵒᵖ ⥤ A) ⥤ Sheaf J A := (equivSmallModel C).transportAndSheafify J
  ((equivSmallModel C).inverse.inducedTopology J) A


/--
Transporting to a small model and sheafifying there is left adjoint to the underlying presheaf
functor
-/
noncomputable
def smallSheafificationAdjunction : smallSheafify J A ⊣ sheafToPresheaf J A :=
  (equivSmallModel C).transportSheafificationAdjunction J _ A


noncomputable instance hasSheafifyEssentiallySmallSite : HasSheafify J A :=
  (equivSmallModel C).hasSheafify J ((equivSmallModel C).inverse.inducedTopology J) A


instance hasSheafComposeEssentiallySmallSite : HasSheafCompose J F :=
  (equivSmallModel C).hasSheafCompose J ((equivSmallModel C).inverse.inducedTopology J) F


instance hasLimitsEssentiallySmallSite
    [HasLimits <| Sheaf ((equivSmallModel C).inverse.inducedTopology J) A] :
    HasLimitsOfSize.{max v₃ w, max v₃ w} <| Sheaf J A :=
  Adjunction.has_limits_of_equivalence ((equivSmallModel C).sheafCongr J
    ((equivSmallModel C).inverse.inducedTopology J) A).functor


instance hasColimitsEssentiallySmallSite
    [HasColimits <| Sheaf ((equivSmallModel C).inverse.inducedTopology J) A] :
    HasColimitsOfSize.{max v₃ w, max v₃ w} <| Sheaf J A :=
  Adjunction.has_colimits_of_equivalence ((equivSmallModel C).sheafCongr J
    ((equivSmallModel C).inverse.inducedTopology J) A).functor


lemma W_inverseImage_whiskeringLeft :
    K.W.inverseImage ((whiskeringLeft Dᵒᵖ Cᵒᵖ A).obj G.op) = J.W := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    inst✝³ : G.IsCoverDense J
    inst✝² : G.Full
    inst✝¹ : G.IsContinuous K J
    inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
    ⊢ Eq (K.W.inverseImage ((CategoryTheory.whiskeringLeft (Opposite D) (Opposite  …
  -/
  ext P Q f
  have h₁ : K.W (A := A) =
    Localization.LeftBousfield.W (· ∈ Set.range (sheafToPresheaf J A ⋙
      ((whiskeringLeft Dᵒᵖ Cᵒᵖ A).obj G.op)).obj) := by
    rw [W_eq_W_range_sheafToPresheaf_obj, ← LeftBousfield.W_isoClosure]
    conv_rhs => rw [← LeftBousfield.W_isoClosure]
    apply congr_arg
    ext P
    constructor
    · rintro ⟨_, ⟨R, rfl⟩, ⟨e⟩⟩
      exact ⟨_, ⟨_, rfl⟩, ⟨e.trans ((sheafToPresheaf _ _).mapIso
        ((G.sheafPushforwardContinuous A K J).objObjPreimageIso R).symm)⟩⟩
    · rintro ⟨_, ⟨R, rfl⟩, ⟨e⟩⟩
      exact ⟨G.op ⋙ R.val, ⟨(G.sheafPushforwardContinuous A K J).obj R, rfl⟩, ⟨e⟩⟩
  have h₂ : ∀ (R : Sheaf J A),
    Function.Bijective (fun (g : G.op ⋙ Q ⟶ G.op ⋙ R.val) ↦ whiskerLeft G.op f ≫ g) ↔
      Function.Bijective (fun (g : Q ⟶ R.val) ↦ f ≫ g) := fun R ↦ by
    rw [← Function.Bijective.of_comp_iff _
      (Functor.whiskerLeft_obj_map_bijective_of_isCoverDense J G Q R.val R.cond)]
    exact Function.Bijective.of_comp_iff'
      (Functor.whiskerLeft_obj_map_bijective_of_isCoverDense J G P R.val R.cond)
        (fun g ↦ f ≫ g)
  /-
    case h
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    inst✝³ : G.IsCoverDense J
    inst✝² : G.Full
    inst✝¹ : G.IsContinuous K J
    inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
    P Q : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P Q
    h₁ : Eq K.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.m …
    h₂ : ∀ (R : CategoryTheory.Sheaf J A), Iff (Function.Bijective fun g => Catego …
    ⊢ Iff (K.W.inverseImage ((CategoryTheory.whiskeringLeft (Opposite D) (Opposite …
  -/
  rw [h₁, J.W_eq_W_range_sheafToPresheaf_obj, MorphismProperty.inverseImage_iff]
  /-
    case h
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    inst✝³ : G.IsCoverDense J
    inst✝² : G.Full
    inst✝¹ : G.IsContinuous K J
    inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
    P Q : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P Q
    h₁ : Eq K.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.m …
    h₂ : ∀ (R : CategoryTheory.Sheaf J A), Iff (Function.Bijective fun g => Catego …
    ⊢ Iff (CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (S …
  -/
  constructor
    /-
      case h.mp
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
      inst✝³ : G.IsCoverDense J
      inst✝² : G.Full
      inst✝¹ : G.IsContinuous K J
      inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      h₁ : Eq K.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.m …
      h₂ : ∀ (R : CategoryTheory.Sheaf J A), Iff (Function.Bijective fun g => Catego …
      ⊢ CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (Set.ra …
    -/
  · rintro h _ ⟨R, rfl⟩
    /-
      case h.mp.intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
      inst✝³ : G.IsCoverDense J
      inst✝² : G.Full
      inst✝¹ : G.IsContinuous K J
      inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      h₁ : Eq K.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.m …
      h₂ : ∀ (R : CategoryTheory.Sheaf J A), Iff (Function.Bijective fun g => Catego …
      h : CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (Set. …
      R : CategoryTheory.Sheaf J A
      ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp f g
    -/
    exact (h₂ R).1 (h _ ⟨R, rfl⟩)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
      inst✝³ : G.IsCoverDense J
      inst✝² : G.Full
      inst✝¹ : G.IsContinuous K J
      inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      h₁ : Eq K.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.m …
      h₂ : ∀ (R : CategoryTheory.Sheaf J A), Iff (Function.Bijective fun g => Catego …
      ⊢ CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (Set.ra …
    -/
  · rintro h _ ⟨R, rfl⟩
    /-
      case h.mpr.intro
      C : Type u₁
      inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
      inst✝³ : G.IsCoverDense J
      inst✝² : G.Full
      inst✝¹ : G.IsContinuous K J
      inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      h₁ : Eq K.W (CategoryTheory.Localization.LeftBousfield.W fun x => Membership.m …
      h₂ : ∀ (R : CategoryTheory.Sheaf J A), Iff (Function.Bijective fun g => Catego …
      h : CategoryTheory.Localization.LeftBousfield.W (fun x => Membership.mem (Set. …
      R : CategoryTheory.Sheaf J A
      ⊢ Function.Bijective fun g => CategoryTheory.CategoryStruct.comp (((CategoryTh …
    -/
    exact (h₂ R).2 (h _ ⟨R, rfl⟩)
    /-
      🎉 no goals
    -/


lemma W_whiskerLeft_iff {P Q : Cᵒᵖ ⥤ A} (f : P ⟶ Q) :
    K.W (whiskerLeft G.op f) ↔ J.W f := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    inst✝³ : G.IsCoverDense J
    inst✝² : G.Full
    inst✝¹ : G.IsContinuous K J
    inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
    P Q : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P Q
    ⊢ Iff (K.W (CategoryTheory.whiskerLeft G.op f)) (J.W f)
  -/
  rw [← W_inverseImage_whiskeringLeft J K G]
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    J : CategoryTheory.GrothendieckTopology C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor D C
    A : Type u₃
    inst✝⁴ : CategoryTheory.Category.{v₃, u₃} A
    inst✝³ : G.IsCoverDense J
    inst✝² : G.Full
    inst✝¹ : G.IsContinuous K J
    inst✝ : (G.sheafPushforwardContinuous A K J).EssSurj
    P Q : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom P Q
    ⊢ Iff (K.W (CategoryTheory.whiskerLeft G.op f)) (K.W.inverseImage ((CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma PreservesSheafification.transport
    [Functor.IsContinuous.{v₄} G K J] [Functor.IsContinuous.{v₃} G K J]
    [(G.sheafPushforwardContinuous B K J).EssSurj]
    [(G.sheafPushforwardContinuous A K J).EssSurj]
    [K.PreservesSheafification F] : J.PreservesSheafification F where
  le P Q f hf := by
    /-
      C : Type u₁
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} A
      B : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} B
      F : CategoryTheory.Functor A B
      inst✝⁶ : G.IsCoverDense J
      inst✝⁵ : G.Full
      inst✝⁴ : G.IsContinuous K J
      inst✝³ : G.IsContinuous K J
      inst✝² : (G.sheafPushforwardContinuous B K J).EssSurj
      inst✝¹ : (G.sheafPushforwardContinuous A K J).EssSurj
      inst✝ : K.PreservesSheafification F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      hf : J.W f
      ⊢ J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) A B).obj F) f
    -/
    rw [← J.W_whiskerLeft_iff (G := G) (K := K)] at hf
    /-
      C : Type u₁
      inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
      J : CategoryTheory.GrothendieckTopology C
      D : Type u₂
      inst✝⁹ : CategoryTheory.Category.{v₂, u₂} D
      K : CategoryTheory.GrothendieckTopology D
      G : CategoryTheory.Functor D C
      A : Type u₃
      inst✝⁸ : CategoryTheory.Category.{v₃, u₃} A
      B : Type u₄
      inst✝⁷ : CategoryTheory.Category.{v₄, u₄} B
      F : CategoryTheory.Functor A B
      inst✝⁶ : G.IsCoverDense J
      inst✝⁵ : G.Full
      inst✝⁴ : G.IsContinuous K J
      inst✝³ : G.IsContinuous K J
      inst✝² : (G.sheafPushforwardContinuous B K J).EssSurj
      inst✝¹ : (G.sheafPushforwardContinuous A K J).EssSurj
      inst✝ : K.PreservesSheafification F
      P Q : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom P Q
      hf : K.W (CategoryTheory.whiskerLeft G.op f)
      ⊢ J.W.inverseImage ((CategoryTheory.whiskeringRight (Opposite C) A B).obj F) f
    -/
    have := K.W_of_preservesSheafification F (whiskerLeft G.op f) hf
    rwa [whiskerRight_left,
      K.W_whiskerLeft_iff (G := G) (J := J) (f := whiskerRight f F)] at this


lemma WEqualsLocallyBijective.transport (hG : CoverPreserving K J G) :
    J.WEqualsLocallyBijective A where
  iff f := by
    rw [← W_whiskerLeft_iff J K G f, ← Presheaf.isLocallyInjective_whisker_iff K J G f hG,
      ← Presheaf.isLocallySurjective_whisker_iff K J G f hG, W_iff_isLocallyBijective]


instance [((equivSmallModel C).inverse.inducedTopology J).WEqualsLocallyBijective A] :
    J.WEqualsLocallyBijective A :=
  WEqualsLocallyBijective.transport J ((equivSmallModel C).inverse.inducedTopology J)
    (equivSmallModel C).inverse (IsDenseSubsite.coverPreserving _ _ _)


instance : PreservesSheafification J F :=
  PreservesSheafification.transport (A := A) J
    ((equivSmallModel C).inverse.inducedTopology J) (equivSmallModel C).inverse B F


