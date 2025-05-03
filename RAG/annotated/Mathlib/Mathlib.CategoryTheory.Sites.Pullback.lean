/-- The pullback functor `Sheaf J A ⥤ Sheaf K A` associated to a functor `G : C ⥤ D` in the
same direction as `G`. -/
def sheafPullback : Sheaf J A ⥤ Sheaf K A :=
  (G.sheafPushforwardContinuous A J K).leftAdjoint


/-- The pullback functor is left adjoint to the pushforward functor. -/
def sheafAdjunctionContinuous :
    G.sheafPullback A J K ⊣ G.sheafPushforwardContinuous A J K :=
  Adjunction.ofIsRightAdjoint (G.sheafPushforwardContinuous A J K)


/-- Construction of the pullback of sheaves using a left Kan extension. -/
def sheafPullback [HasWeakSheafify K A] : Sheaf J A ⥤ Sheaf K A :=
  sheafToPresheaf J A ⋙ G.op.lan ⋙ presheafToSheaf K A


/-- The constructed `sheafPullback G A J K` is left adjoint
to `G.sheafPushforwardContinuous A J K`. -/
def sheafAdjunctionContinuous [Functor.IsContinuous.{v₁} G J K] [HasWeakSheafify K A] :
    sheafPullback G A J K ⊣ G.sheafPushforwardContinuous A J K :=
  ((G.op.lanAdjunction A).comp (sheafificationAdjunction K A)).restrictFullyFaithful
    (fullyFaithfulSheafToPresheaf J A) (Functor.FullyFaithful.id _) (Iso.refl _) (Iso.refl _)


instance [HasWeakSheafify K A] :
    (G.sheafPushforwardContinuous A J K).IsRightAdjoint :=
  (sheafAdjunctionContinuous G A J K).isRightAdjoint


/-- The constructed pullback of sheaves is isomorphic to the abstract one. -/
def sheafPullbackIso [HasWeakSheafify K A] :
    Functor.sheafPullback G A J K ≅ sheafPullback G A J K :=
  Adjunction.leftAdjointUniq (Functor.sheafAdjunctionContinuous G A J K)
    (sheafAdjunctionContinuous G A J K)


instance : PreservesFiniteLimits (sheafPullback G A J K) := by
  have : PreservesFiniteLimits (G.op.lan ⋙ presheafToSheaf K A) :=
    comp_preservesFiniteLimits _ _
  /-
    C : Type u₂
    inst✝⁸ : CategoryTheory.Category.{v₂, u₂} C
    D : Type u₃
    inst✝⁷ : CategoryTheory.Category.{v₃, u₃} D
    G : CategoryTheory.Functor C D
    A : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁵ : G.IsContinuous J K
    inst✝⁴ : ∀ (F : CategoryTheory.Functor (Opposite C) A), G.op.HasLeftKanExtensi …
    inst✝³ : CategoryTheory.RepresentablyFlat G
    inst✝² : CategoryTheory.HasSheafify K A
    inst✝¹ : CategoryTheory.HasSheafify J A
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits G.op.lan
    this : CategoryTheory.Limits.PreservesFiniteLimits (G.op.lan.comp (CategoryThe …
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (CategoryTheory.Functor.sheafPul …
  -/
  apply comp_preservesFiniteLimits
  /-
    🎉 no goals
  -/


instance preservesFiniteLimits : PreservesFiniteLimits (Functor.sheafPullback G A J K) :=
  preservesFiniteLimits_of_natIso (sheafPullbackIso G A J K).symm


attribute [local instance] reflectsLimits_of_reflectsIsomorphisms in
instance [RepresentablyFlat G] : PreservesFiniteLimits (G.sheafPullback A J K) := by
  /-
    C : Type v₁
    inst✝¹⁰ : CategoryTheory.SmallCategory C
    D : Type v₁
    inst✝⁹ : CategoryTheory.SmallCategory D
    G : CategoryTheory.Functor C D
    A : Type u₁
    inst✝⁸ : CategoryTheory.Category.{v₁, u₁} A
    J : CategoryTheory.GrothendieckTopology C
    K : CategoryTheory.GrothendieckTopology D
    inst✝⁷ : CategoryTheory.ConcreteCategory A
    inst✝⁶ : CategoryTheory.Limits.PreservesLimits (CategoryTheory.forget A)
    inst✝⁵ : CategoryTheory.Limits.HasColimits A
    inst✝⁴ : CategoryTheory.Limits.HasLimits A
    inst✝³ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forge …
    inst✝² : (CategoryTheory.forget A).ReflectsIsomorphisms
    inst✝¹ : G.IsContinuous J K
    inst✝ : CategoryTheory.RepresentablyFlat G
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits (G.sheafPullback A J K)
  -/
  apply sheafPullbackConstruction.preservesFiniteLimits
  /-
    🎉 no goals
  -/


