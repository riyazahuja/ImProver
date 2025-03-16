/-- The image of a 1-pre-hypercover by a functor. -/
@[simps]
def map : PreOneHypercover (F.obj X) where
  I₀ := E.I₀
  X i := F.obj (E.X i)
  f i := F.map (E.f i)
  I₁ := E.I₁
  Y _ _ j := F.obj (E.Y j)
  p₁ _ _ j := F.map (E.p₁ j)
  p₂ _ _ j := F.map (E.p₂ j)
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  E✝ : Type u₃
                  inst✝ : CategoryTheory.Category.{v₃, u₃} E✝
                  X : C
                  E : CategoryTheory.PreOneHypercover X
                  F : CategoryTheory.Functor C D
                  x✝¹ x✝ : E.I₀
                  j : E.I₁ x✝¹ x✝
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x x_1 j => F.map (E.p₁ j)) x✝¹  …
                -/
  w _ _ j := by simpa using F.congr_map (E.w j)
                /-
                  🎉 no goals
                -/


/-- If `F : C ⥤ D`, `P : Dᵒᵖ ⥤ A` and `E` is a 1-pre-hypercover of an object of `X`,
then `(E.map F).multifork P` is a limit iff `E.multifork (F.op ⋙ P)` is a limit. -/
def isLimitMapMultiforkEquiv {A : Type u} [Category.{t} A] (P : Dᵒᵖ ⥤ A) :
                                                                             /-
                                                                               C : Type u₁
                                                                               inst✝³ : CategoryTheory.Category.{v₁, u₁} C
                                                                               D : Type u₂
                                                                               inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                                                               E✝ : Type u₃
                                                                               inst✝¹ : CategoryTheory.Category.{v₃, u₃} E✝
                                                                               X : C
                                                                               E : CategoryTheory.PreOneHypercover X
                                                                               F : CategoryTheory.Functor C D
                                                                               A : Type u
                                                                               inst✝ : CategoryTheory.Category.{t, u} A
                                                                               P : CategoryTheory.Functor (Opposite D) A
                                                                               ⊢ Equiv (CategoryTheory.Limits.IsLimit ((E.map F).multifork P)) (CategoryTheor …
                                                                             -/
    IsLimit ((E.map F).multifork P) ≃ IsLimit (E.multifork (F.op ⋙ P)) := by rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


/-- A 1-hypercover in `C` is preserved by a functor `F : C ⥤ D` if the mapped 1-pre-hypercover
in `D` is a 1-hypercover for the given topology on `D`. -/
class IsPreservedBy (F : C ⥤ D) (K : GrothendieckTopology D) : Prop where
  mem₀ : (E.toPreOneHypercover.map F).sieve₀ ∈ K (F.obj X)
  mem₁ (i₁ i₂ : E.I₀) ⦃W : D⦄ (p₁ : W ⟶ F.obj (E.X i₁)) (p₂ : W ⟶ F.obj (E.X i₂))
    (w : p₁ ≫ F.map (E.f i₁) = p₂ ≫ F.map (E.f i₂)) :
      (E.toPreOneHypercover.map F).sieve₁ p₁ p₂ ∈ K W


/-- Given a 1-hypercover `E : J.OneHypercover X` of an object of `C`, a functor `F : C ⥤ D`
such that `E.IsPreversedBy F K` for a Grothendieck topology `K` on `D`, this is
the image of `E` by `F`, as a 1-hypercover of `F.obj X` for `K`. -/
@[simps toPreOneHypercover]
def map (F : C ⥤ D) (K : GrothendieckTopology D) [E.IsPreservedBy F K] :
    K.OneHypercover (F.obj X) where
  toPreOneHypercover := E.toPreOneHypercover.map F
  mem₀ := IsPreservedBy.mem₀
  mem₁ := IsPreservedBy.mem₁


instance : E.IsPreservedBy (𝟭 C) J where
  mem₀ := E.mem₀
  mem₁ := E.mem₁


/-- The condition that a functor `F : C ⥤ D` sends 1-hypercovers for
`J : GrothendieckTopology C` to 1-hypercovers for `K : GrothendieckTopology D`. -/
abbrev PreservesOneHypercovers :=
  ∀ {X : C} (E : GrothendieckTopology.OneHypercover.{w} J X), E.IsPreservedBy F K


/-- A functor `F` is continuous if the precomposition with `F.op` sends sheaves of `Type t`
to sheaves. -/
class IsContinuous : Prop where
  op_comp_isSheaf_of_types (G : Sheaf K (Type t)) : Presieve.IsSheaf J (F.op ⋙ G.val)


lemma op_comp_isSheaf_of_types [Functor.IsContinuous.{t} F J K] (G : Sheaf K (Type t)) :
    Presieve.IsSheaf J (F.op ⋙ G.val) :=
  Functor.IsContinuous.op_comp_isSheaf_of_types _


@[deprecated (since := "2024-11-26")] alias op_comp_isSheafOfTypes := op_comp_isSheaf_of_types


lemma op_comp_isSheaf [Functor.IsContinuous.{t} F J K] (G : Sheaf K A) :
    Presheaf.IsSheaf J (F.op ⋙ G.val) :=
  fun T => F.op_comp_isSheaf_of_types J K ⟨_, (isSheaf_iff_isSheaf_of_type _ _).2 (G.cond T)⟩


lemma isContinuous_of_iso {F₁ F₂ : C ⥤ D} (e : F₁ ≅ F₂)
    (J : GrothendieckTopology C) (K : GrothendieckTopology D)
    [Functor.IsContinuous.{t} F₁ J K] : Functor.IsContinuous.{t} F₂ J K where
  op_comp_isSheaf_of_types G :=
    Presieve.isSheaf_iso J (isoWhiskerRight (NatIso.op e.symm) _)
      (F₁.op_comp_isSheaf_of_types J K G)


instance isContinuous_id : Functor.IsContinuous.{w} (𝟭 C) J J where
  op_comp_isSheaf_of_types G := (isSheaf_iff_isSheaf_of_type _ _).1 G.2


lemma isContinuous_comp (F₁ : C ⥤ D) (F₂ : D ⥤ E) (J : GrothendieckTopology C)
    (K : GrothendieckTopology D) (L : GrothendieckTopology E)
    [Functor.IsContinuous.{t} F₁ J K] [Functor.IsContinuous.{t} F₂ K L] :
    Functor.IsContinuous.{t} (F₁ ⋙ F₂) J L where
  op_comp_isSheaf_of_types G :=
    F₁.op_comp_isSheaf_of_types J K
      ⟨_,(isSheaf_iff_isSheaf_of_type _ _).2 (F₂.op_comp_isSheaf_of_types K L G)⟩


lemma isContinuous_comp' {F₁ : C ⥤ D} {F₂ : D ⥤ E} {F₁₂ : C ⥤ E}
    (e : F₁ ⋙ F₂ ≅ F₁₂) (J : GrothendieckTopology C)
    (K : GrothendieckTopology D) (L : GrothendieckTopology E)
    [Functor.IsContinuous.{t} F₁ J K] [Functor.IsContinuous.{t} F₂ K L] :
    Functor.IsContinuous.{t} F₁₂ J L := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F₁ : CategoryTheory.Functor C D
    F₂ : CategoryTheory.Functor D E
    F₁₂ : CategoryTheory.Functor C E
    e : CategoryTheory.Iso (F₁.comp F₂) F₁₂
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    L : CategoryTheory.GrothendieckTopology E
    inst✝¹ : F₁.IsContinuous J K
    inst✝ : F₂.IsContinuous K L
    ⊢ F₁₂.IsContinuous J L
  -/
  have := Functor.isContinuous_comp F₁ F₂ J K L
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝² : CategoryTheory.Category.{v₃, u₃} E
    F₁ : CategoryTheory.Functor C D
    F₂ : CategoryTheory.Functor D E
    F₁₂ : CategoryTheory.Functor C E
    e : CategoryTheory.Iso (F₁.comp F₂) F₁₂
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    L : CategoryTheory.GrothendieckTopology E
    inst✝¹ : F₁.IsContinuous J K
    inst✝ : F₂.IsContinuous K L
    this : (F₁.comp F₂).IsContinuous J L
    ⊢ F₁₂.IsContinuous J L
  -/
  apply Functor.isContinuous_of_iso e
  /-
    🎉 no goals
  -/


lemma op_comp_isSheaf_of_preservesOneHypercovers
    [PreservesOneHypercovers.{w} F J K] [GrothendieckTopology.IsGeneratedByOneHypercovers.{w} J]
    (P : Dᵒᵖ ⥤ A) (hP : Presheaf.IsSheaf K P) :
    Presheaf.IsSheaf J (F.op ⋙ P) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    A : Type u
    inst✝² : CategoryTheory.Category.{t, u} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝¹ : F.PreservesOneHypercovers J K
    inst✝ : J.IsGeneratedByOneHypercovers
    P : CategoryTheory.Functor (Opposite D) A
    hP : CategoryTheory.Presheaf.IsSheaf K P
    ⊢ CategoryTheory.Presheaf.IsSheaf J (F.op.comp P)
  -/
  rw [Presheaf.isSheaf_iff_of_isGeneratedByOneHypercovers.{w}]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    A : Type u
    inst✝² : CategoryTheory.Category.{t, u} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝¹ : F.PreservesOneHypercovers J K
    inst✝ : J.IsGeneratedByOneHypercovers
    P : CategoryTheory.Functor (Opposite D) A
    hP : CategoryTheory.Presheaf.IsSheaf K P
    ⊢ ∀ ⦃X : C⦄ (E : J.OneHypercover X), Nonempty (CategoryTheory.Limits.IsLimit ( …
  -/
  intro X E
  exact ⟨(E.toPreOneHypercover.isLimitMapMultiforkEquiv F P)
    ((E.map F K).isLimitMultifork ⟨P, hP⟩)⟩


lemma isContinuous_of_preservesOneHypercovers
    [PreservesOneHypercovers.{w} F J K] [GrothendieckTopology.IsGeneratedByOneHypercovers.{w} J] :
    IsContinuous.{t} F J K where
  op_comp_isSheaf_of_types := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      inst✝¹ : F.PreservesOneHypercovers J K
      inst✝ : J.IsGeneratedByOneHypercovers
      ⊢ ∀ (G : CategoryTheory.Sheaf K (Type t)), CategoryTheory.Presieve.IsSheaf J ( …
    -/
    rintro ⟨P, hP⟩
    /-
      case mk
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      inst✝¹ : F.PreservesOneHypercovers J K
      inst✝ : J.IsGeneratedByOneHypercovers
      P : CategoryTheory.Functor (Opposite D) (Type t)
      hP : CategoryTheory.Presheaf.IsSheaf K P
      ⊢ CategoryTheory.Presieve.IsSheaf J (F.op.comp { val := P, cond := hP }.val)
    -/
    rw [← isSheaf_iff_isSheaf_of_type]
    /-
      case mk
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      F : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      inst✝¹ : F.PreservesOneHypercovers J K
      inst✝ : J.IsGeneratedByOneHypercovers
      P : CategoryTheory.Functor (Opposite D) (Type t)
      hP : CategoryTheory.Presheaf.IsSheaf K P
      ⊢ CategoryTheory.Presheaf.IsSheaf J (F.op.comp { val := P, cond := hP }.val)
    -/
    exact F.op_comp_isSheaf_of_preservesOneHypercovers J K P hP
    /-
      🎉 no goals
    -/


instance [PreservesOneHypercovers.{max u₁ v₁} F J K] :
    IsContinuous.{t} F J K :=
  isContinuous_of_preservesOneHypercovers.{max u₁ v₁} F J K


/-- The induced functor `Sheaf K A ⥤ Sheaf J A` given by `F.op ⋙ _`
if `F` is a continuous functor.
-/
@[simps!]
def sheafPushforwardContinuous : Sheaf K A ⥤ Sheaf J A where
  obj ℱ := ⟨F.op ⋙ ℱ.val, F.op_comp_isSheaf J K ℱ⟩
  map f := ⟨((whiskeringLeft _ _ _).obj F.op).map f.val⟩


/-- The functor `F.sheafPushforwardContinuous A J K : Sheaf K A ⥤ Sheaf J A`
is induced by the precomposition with `F.op`. -/
@[simps!]
def sheafPushforwardContinuousCompSheafToPresheafIso :
    F.sheafPushforwardContinuous A J K ⋙ sheafToPresheaf J A ≅
      sheafToPresheaf K A ⋙ (whiskeringLeft _ _ _).obj F.op := Iso.refl _


