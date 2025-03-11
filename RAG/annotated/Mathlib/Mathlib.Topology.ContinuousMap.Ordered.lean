instance partialOrder [PartialOrder β] : PartialOrder C(α, β) :=
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          inst✝² : TopologicalSpace α
                                                          inst✝¹ : TopologicalSpace β
                                                          inst✝ : PartialOrder β
                                                          f g : ContinuousMap α β
                                                          x✝ : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
                                                          ⊢ Eq f g
                                                        -/
  PartialOrder.lift (fun f => f.toFun) (fun f g _ => by aesop)
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem le_def [PartialOrder β] {f g : C(α, β)} : f ≤ g ↔ ∀ a, f a ≤ g a :=
  Pi.le_def


theorem lt_def [PartialOrder β] {f g : C(α, β)} : f < g ↔ (∀ a, f a ≤ g a) ∧ ∃ a, f a < g a :=
  Pi.lt_def


instance sup : Max C(α, β) where max f g := { toFun := fun a ↦ f a ⊔ g a }


@[simp, norm_cast] lemma coe_sup (f g : C(α, β)) : ⇑(f ⊔ g) = ⇑f ⊔ g := rfl


@[simp] lemma sup_apply (f g : C(α, β)) (a : α) : (f ⊔ g) a = f a ⊔ g a := rfl


instance semilatticeSup : SemilatticeSup C(α, β) :=
  DFunLike.coe_injective.semilatticeSup _ fun _ _ ↦ rfl


lemma sup'_apply {ι : Type*} {s : Finset ι} (H : s.Nonempty) (f : ι → C(α, β)) (a : α) :
    s.sup' H f a = s.sup' H fun i ↦ f i a :=
  Finset.comp_sup'_eq_sup'_comp H (fun g : C(α, β) ↦ g a) fun _ _ ↦ rfl


@[simp, norm_cast]
lemma coe_sup' {ι : Type*} {s : Finset ι} (H : s.Nonempty) (f : ι → C(α, β)) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    inst✝³ : TopologicalSpace α
                                                    inst✝² : TopologicalSpace β
                                                    inst✝¹ : SemilatticeSup β
                                                    inst✝ : ContinuousSup β
                                                    ι : Type u_3
                                                    s : Finset ι
                                                    H : s.Nonempty
                                                    f : ι → ContinuousMap α β
                                                    ⊢ Eq (⇑(s.sup' H f)) (s.sup' H fun i => ⇑(f i))
                                                  -/
    ⇑(s.sup' H f) = s.sup' H fun i ↦ ⇑(f i) := by ext; simp [sup'_apply]
                                                       /-
                                                         🎉 no goals
                                                       -/


instance inf : Min C(α, β) where min f g := { toFun := fun a ↦ f a ⊓ g a }


@[simp, norm_cast] lemma coe_inf (f g : C(α, β)) : ⇑(f ⊓ g) = ⇑f ⊓ g := rfl


@[simp] lemma inf_apply (f g : C(α, β)) (a : α) : (f ⊓ g) a = f a ⊓ g a := rfl


instance semilatticeInf : SemilatticeInf C(α, β) :=
  DFunLike.coe_injective.semilatticeInf _ fun _ _ ↦ rfl


lemma inf'_apply {ι : Type*} {s : Finset ι} (H : s.Nonempty) (f : ι → C(α, β)) (a : α) :
    s.inf' H f a = s.inf' H fun i ↦ f i a :=
  Finset.comp_inf'_eq_inf'_comp H (fun g : C(α, β) ↦ g a) fun _ _ ↦ rfl


@[simp, norm_cast]
lemma coe_inf' {ι : Type*} {s : Finset ι} (H : s.Nonempty) (f : ι → C(α, β)) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    inst✝³ : TopologicalSpace α
                                                    inst✝² : TopologicalSpace β
                                                    inst✝¹ : SemilatticeInf β
                                                    inst✝ : ContinuousInf β
                                                    ι : Type u_3
                                                    s : Finset ι
                                                    H : s.Nonempty
                                                    f : ι → ContinuousMap α β
                                                    ⊢ Eq (⇑(s.inf' H f)) (s.inf' H fun i => ⇑(f i))
                                                  -/
    ⇑(s.inf' H f) = s.inf' H fun i ↦ ⇑(f i) := by ext; simp [inf'_apply]
                                                       /-
                                                         🎉 no goals
                                                       -/


instance [Lattice β] [TopologicalLattice β] : Lattice C(α, β) :=
  DFunLike.coe_injective.lattice _ (fun _ _ ↦ rfl) fun _ _ ↦ rfl

-- TODO transfer this lattice structure to `BoundedContinuousFunction`


/-- Extend a continuous function `f : C(Set.Icc a b, β)` to a function `f : C(α, β)`. -/
def IccExtend (f : C(Set.Icc a b, β)) : C(α, β) where
  toFun := Set.IccExtend h f


@[simp]
theorem coe_IccExtend (f : C(Set.Icc a b, β)) :
    ((IccExtend h f : C(α, β)) : α → β) = Set.IccExtend h f :=
  rfl


