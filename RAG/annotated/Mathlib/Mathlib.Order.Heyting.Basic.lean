instance Prod.instHImp [HImp α] [HImp β] : HImp (α × β) :=
  ⟨fun a b => (a.1 ⇨ b.1, a.2 ⇨ b.2)⟩


instance Prod.instHNot [HNot α] [HNot β] : HNot (α × β) :=
  ⟨fun a => (￢a.1, ￢a.2)⟩


instance Prod.instSDiff [SDiff α] [SDiff β] : SDiff (α × β) :=
  ⟨fun a b => (a.1 \ b.1, a.2 \ b.2)⟩


instance Prod.instHasCompl [HasCompl α] [HasCompl β] : HasCompl (α × β) :=
  ⟨fun a => (a.1ᶜ, a.2ᶜ)⟩


@[simp]
theorem fst_himp [HImp α] [HImp β] (a b : α × β) : (a ⇨ b).1 = a.1 ⇨ b.1 :=
  rfl


@[simp]
theorem snd_himp [HImp α] [HImp β] (a b : α × β) : (a ⇨ b).2 = a.2 ⇨ b.2 :=
  rfl


@[simp]
theorem fst_hnot [HNot α] [HNot β] (a : α × β) : (￢a).1 = ￢a.1 :=
  rfl


@[simp]
theorem snd_hnot [HNot α] [HNot β] (a : α × β) : (￢a).2 = ￢a.2 :=
  rfl


@[simp]
theorem fst_sdiff [SDiff α] [SDiff β] (a b : α × β) : (a \ b).1 = a.1 \ b.1 :=
  rfl


@[simp]
theorem snd_sdiff [SDiff α] [SDiff β] (a b : α × β) : (a \ b).2 = a.2 \ b.2 :=
  rfl


@[simp]
theorem fst_compl [HasCompl α] [HasCompl β] (a : α × β) : aᶜ.1 = a.1ᶜ :=
  rfl


@[simp]
theorem snd_compl [HasCompl α] [HasCompl β] (a : α × β) : aᶜ.2 = a.2ᶜ :=
  rfl


instance [∀ i, HImp (π i)] : HImp (∀ i, π i) :=
  ⟨fun a b i => a i ⇨ b i⟩


instance [∀ i, HNot (π i)] : HNot (∀ i, π i) :=
  ⟨fun a i => ￢a i⟩


theorem himp_def [∀ i, HImp (π i)] (a b : ∀ i, π i) : a ⇨ b = fun i => a i ⇨ b i :=
  rfl


theorem hnot_def [∀ i, HNot (π i)] (a : ∀ i, π i) : ￢a = fun i => ￢a i :=
  rfl


@[simp]
theorem himp_apply [∀ i, HImp (π i)] (a b : ∀ i, π i) (i : ι) : (a ⇨ b) i = a i ⇨ b i :=
  rfl


@[simp]
theorem hnot_apply [∀ i, HNot (π i)] (a : ∀ i, π i) (i : ι) : (￢a) i = ￢a i :=
  rfl


/-- A generalized Heyting algebra is a lattice with an additional binary operation `⇨` called
Heyting implication such that `(a ⇨ ·)` is right adjoint to `(a ⊓ ·)`.

 This generalizes `HeytingAlgebra` by not requiring a bottom element. -/
class GeneralizedHeytingAlgebra (α : Type*) extends Lattice α, OrderTop α, HImp α where
  /-- `(a ⇨ ·)` is right adjoint to `(a ⊓ ·)` -/
  le_himp_iff (a b c : α) : a ≤ b ⇨ c ↔ a ⊓ b ≤ c


/-- A generalized co-Heyting algebra is a lattice with an additional binary
difference operation `\` such that `(· \ a)` is right adjoint to `(· ⊔ a)`.

This generalizes `CoheytingAlgebra` by not requiring a top element. -/
class GeneralizedCoheytingAlgebra (α : Type*) extends Lattice α, OrderBot α, SDiff α where
  /-- `(· \ a)` is right adjoint to `(· ⊔ a)` -/
  sdiff_le_iff (a b c : α) : a \ b ≤ c ↔ a ≤ b ⊔ c


/-- A Heyting algebra is a bounded lattice with an additional binary operation `⇨` called Heyting
implication such that `(a ⇨ ·)` is right adjoint to `(a ⊓ ·)`. -/
class HeytingAlgebra (α : Type*) extends GeneralizedHeytingAlgebra α, OrderBot α, HasCompl α where
  /-- `aᶜ` is defined as `a ⇨ ⊥` -/
  himp_bot (a : α) : a ⇨ ⊥ = aᶜ


/-- A co-Heyting algebra is a bounded lattice with an additional binary difference operation `\`
such that `(· \ a)` is right adjoint to `(· ⊔ a)`. -/
class CoheytingAlgebra (α : Type*) extends GeneralizedCoheytingAlgebra α, OrderTop α, HNot α where
  /-- `⊤ \ a` is `￢a` -/
  top_sdiff (a : α) : ⊤ \ a = ￢a


/-- A bi-Heyting algebra is a Heyting algebra that is also a co-Heyting algebra. -/
class BiheytingAlgebra (α : Type*) extends HeytingAlgebra α, SDiff α, HNot α where
  /-- `(· \ a)` is right adjoint to `(· ⊔ a)` -/
  sdiff_le_iff (a b c : α) : a \ b ≤ c ↔ a ≤ b ⊔ c
  /-- `⊤ \ a` is `￢a` -/
  top_sdiff (a : α) : ⊤ \ a = ￢a

-- See note [lower instance priority]

instance (priority := 100) HeytingAlgebra.toBoundedOrder [HeytingAlgebra α] : BoundedOrder α :=
  { bot_le := ‹HeytingAlgebra α›.bot_le }

-- See note [lower instance priority]

instance (priority := 100) CoheytingAlgebra.toBoundedOrder [CoheytingAlgebra α] : BoundedOrder α :=
  { ‹CoheytingAlgebra α› with }

-- See note [lower instance priority]

instance (priority := 100) BiheytingAlgebra.toCoheytingAlgebra [BiheytingAlgebra α] :
    CoheytingAlgebra α :=
  { ‹BiheytingAlgebra α› with }

-- See note [reducible non-instances]

/-- Construct a Heyting algebra from the lattice structure and Heyting implication alone. -/
abbrev HeytingAlgebra.ofHImp [DistribLattice α] [BoundedOrder α] (himp : α → α → α)
    (le_himp_iff : ∀ a b c, a ≤ himp b c ↔ a ⊓ b ≤ c) : HeytingAlgebra α :=
  { ‹DistribLattice α›, ‹BoundedOrder α› with
    himp,
    compl := fun a => himp a ⊥,
    le_himp_iff,
    himp_bot := fun _ => rfl }

-- See note [reducible non-instances]

/-- Construct a Heyting algebra from the lattice structure and complement operator alone. -/
abbrev HeytingAlgebra.ofCompl [DistribLattice α] [BoundedOrder α] (compl : α → α)
    (le_himp_iff : ∀ a b c, a ≤ compl b ⊔ c ↔ a ⊓ b ≤ c) : HeytingAlgebra α where
  himp := (compl · ⊔ ·)
  compl := compl
  le_himp_iff := le_himp_iff
  himp_bot _ := sup_bot_eq _

-- See note [reducible non-instances]

/-- Construct a co-Heyting algebra from the lattice structure and the difference alone. -/
abbrev CoheytingAlgebra.ofSDiff [DistribLattice α] [BoundedOrder α] (sdiff : α → α → α)
    (sdiff_le_iff : ∀ a b c, sdiff a b ≤ c ↔ a ≤ b ⊔ c) : CoheytingAlgebra α :=
  { ‹DistribLattice α›, ‹BoundedOrder α› with
    sdiff,
    hnot := fun a => sdiff ⊤ a,
    sdiff_le_iff,
    top_sdiff := fun _ => rfl }

-- See note [reducible non-instances]

/-- Construct a co-Heyting algebra from the difference and Heyting negation alone. -/
abbrev CoheytingAlgebra.ofHNot [DistribLattice α] [BoundedOrder α] (hnot : α → α)
    (sdiff_le_iff : ∀ a b c, a ⊓ hnot b ≤ c ↔ a ≤ b ⊔ c) : CoheytingAlgebra α where
  sdiff a b := a ⊓ hnot b
  hnot := hnot
  sdiff_le_iff := sdiff_le_iff
  top_sdiff _ := top_inf_eq _


/-- `p → q → r ↔ p ∧ q → r` -/
@[simp]
theorem le_himp_iff : a ≤ b ⇨ c ↔ a ⊓ b ≤ c :=
  GeneralizedHeytingAlgebra.le_himp_iff _ _ _


/-- `p → q → r ↔ q ∧ p → r` -/
                                                   /-
                                                     α : Type u_2
                                                     inst✝ : GeneralizedHeytingAlgebra α
                                                     a b c : α
                                                     ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le (Min.min b a) c)
                                                   -/
theorem le_himp_iff' : a ≤ b ⇨ c ↔ b ⊓ a ≤ c := by rw [le_himp_iff, inf_comm]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `p → q → r ↔ q → p → r` -/
                                                   /-
                                                     α : Type u_2
                                                     inst✝ : GeneralizedHeytingAlgebra α
                                                     a b c : α
                                                     ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le b (HImp.himp a c))
                                                   -/
theorem le_himp_comm : a ≤ b ⇨ c ↔ b ≤ a ⇨ c := by rw [le_himp_iff, le_himp_iff']
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `p → q → p` -/
theorem le_himp : a ≤ b ⇨ a :=
  le_himp_iff.2 inf_le_left


/-- `p → p → q ↔ p → q` -/
                                                   /-
                                                     α : Type u_2
                                                     inst✝ : GeneralizedHeytingAlgebra α
                                                     a b : α
                                                     ⊢ Iff (LE.le a (HImp.himp a b)) (LE.le a b)
                                                   -/
theorem le_himp_iff_left : a ≤ a ⇨ b ↔ a ≤ b := by rw [le_himp_iff, inf_idem]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `p → p` -/
@[simp]
theorem himp_self : a ⇨ a = ⊤ :=
  top_le_iff.1 <| le_himp_iff.2 inf_le_right


/-- `(p → q) ∧ p → q` -/
theorem himp_inf_le : (a ⇨ b) ⊓ a ≤ b :=
  le_himp_iff.1 le_rfl


/-- `p ∧ (p → q) → q` -/
                                            /-
                                              α : Type u_2
                                              inst✝ : GeneralizedHeytingAlgebra α
                                              a b : α
                                              ⊢ LE.le (Min.min a (HImp.himp a b)) b
                                            -/
theorem inf_himp_le : a ⊓ (a ⇨ b) ≤ b := by rw [inf_comm, ← le_himp_iff]
                                            /-
                                              🎉 no goals
                                            -/


/-- `p ∧ (p → q) ↔ p ∧ q` -/
@[simp]
theorem inf_himp (a b : α) : a ⊓ (a ⇨ b) = a ⊓ b :=
                                        /-
                                          α : Type u_2
                                          inst✝ : GeneralizedHeytingAlgebra α
                                          a b : α
                                          ⊢ LE.le (Min.min a (HImp.himp a b)) b
                                        -/
  le_antisymm (le_inf inf_le_left <| by rw [inf_comm, ← le_himp_iff]) <| inf_le_inf_left _ le_himp
                                        /-
                                          🎉 no goals
                                        -/


/-- `(p → q) ∧ p ↔ q ∧ p` -/
@[simp]
                                                            /-
                                                              α : Type u_2
                                                              inst✝ : GeneralizedHeytingAlgebra α
                                                              a b : α
                                                              ⊢ Eq (Min.min (HImp.himp a b) a) (Min.min b a)
                                                            -/
theorem himp_inf_self (a b : α) : (a ⇨ b) ⊓ a = b ⊓ a := by rw [inf_comm, inf_himp, inf_comm]
                                                            /-
                                                              🎉 no goals
                                                            -/


/-- The **deduction theorem** in the Heyting algebra model of intuitionistic logic:
an implication holds iff the conclusion follows from the hypothesis. -/
@[simp]
                                                  /-
                                                    α : Type u_2
                                                    inst✝ : GeneralizedHeytingAlgebra α
                                                    a b : α
                                                    ⊢ Iff (Eq (HImp.himp a b) Top.top) (LE.le a b)
                                                  -/
theorem himp_eq_top_iff : a ⇨ b = ⊤ ↔ a ≤ b := by rw [← top_le_iff, le_himp_iff, top_inf_eq]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- `p → true`, `true → p ↔ p` -/
@[simp]
theorem himp_top : a ⇨ ⊤ = ⊤ :=
  himp_eq_top_iff.2 le_top


@[simp]
theorem top_himp : ⊤ ⇨ a = a :=
                                  /-
                                    α : Type u_2
                                    inst✝ : GeneralizedHeytingAlgebra α
                                    a b : α
                                    ⊢ Iff (LE.le b (HImp.himp Top.top a)) (LE.le b a)
                                  -/
  eq_of_forall_le_iff fun b => by rw [le_himp_iff, inf_top_eq]
                                  /-
                                    🎉 no goals
                                  -/


/-- `p → q → r ↔ p ∧ q → r` -/
theorem himp_himp (a b c : α) : a ⇨ b ⇨ c = a ⊓ b ⇨ c :=
                                  /-
                                    α : Type u_2
                                    inst✝ : GeneralizedHeytingAlgebra α
                                    a b c d : α
                                    ⊢ Iff (LE.le d (HImp.himp a (HImp.himp b c))) (LE.le d (HImp.himp (Min.min a b …
                                  -/
  eq_of_forall_le_iff fun d => by simp_rw [le_himp_iff, inf_assoc]
                                  /-
                                    🎉 no goals
                                  -/


/-- `(q → r) → (p → q) → q → r` -/
theorem himp_le_himp_himp_himp : b ⇨ c ≤ (a ⇨ b) ⇨ a ⇨ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ LE.le (HImp.himp b c) (HImp.himp (HImp.himp a b) (HImp.himp a c))
  -/
  rw [le_himp_iff, le_himp_iff, inf_assoc, himp_inf_self, ← inf_assoc, himp_inf_self, inf_assoc]
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ LE.le (Min.min c (Min.min b a)) c
  -/
  exact inf_le_left
  /-
    🎉 no goals
  -/


@[simp]
theorem himp_inf_himp_inf_le : (b ⇨ c) ⊓ (a ⇨ b) ⊓ a ≤ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ LE.le (Min.min (Min.min (HImp.himp b c) (HImp.himp a b)) a) c
  -/
  simpa using @himp_le_himp_himp_himp
  /-
    🎉 no goals
  -/


/-- `p → q → r ↔ q → p → r` -/
                                                                 /-
                                                                   α : Type u_2
                                                                   inst✝ : GeneralizedHeytingAlgebra α
                                                                   a b c : α
                                                                   ⊢ Eq (HImp.himp a (HImp.himp b c)) (HImp.himp b (HImp.himp a c))
                                                                 -/
theorem himp_left_comm (a b c : α) : a ⇨ b ⇨ c = b ⇨ a ⇨ c := by simp_rw [himp_himp, inf_comm]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
                                            /-
                                              α : Type u_2
                                              inst✝ : GeneralizedHeytingAlgebra α
                                              a b : α
                                              ⊢ Eq (HImp.himp b (HImp.himp b a)) (HImp.himp b a)
                                            -/
theorem himp_idem : b ⇨ b ⇨ a = b ⇨ a := by rw [himp_himp, inf_idem]
                                            /-
                                              🎉 no goals
                                            -/


theorem himp_inf_distrib (a b c : α) : a ⇨ b ⊓ c = (a ⇨ b) ⊓ (a ⇨ c) :=
                                  /-
                                    α : Type u_2
                                    inst✝ : GeneralizedHeytingAlgebra α
                                    a b c d : α
                                    ⊢ Iff (LE.le d (HImp.himp a (Min.min b c))) (LE.le d (Min.min (HImp.himp a b)  …
                                  -/
  eq_of_forall_le_iff fun d => by simp_rw [le_himp_iff, le_inf_iff, le_himp_iff]
                                  /-
                                    🎉 no goals
                                  -/


theorem sup_himp_distrib (a b c : α) : a ⊔ b ⇨ c = (a ⇨ c) ⊓ (b ⇨ c) :=
  eq_of_forall_le_iff fun d => by
    /-
      α : Type u_2
      inst✝ : GeneralizedHeytingAlgebra α
      a b c d : α
      ⊢ Iff (LE.le d (HImp.himp (Max.max a b) c)) (LE.le d (Min.min (HImp.himp a c)  …
    -/
    rw [le_inf_iff, le_himp_comm, sup_le_iff]
    /-
      α : Type u_2
      inst✝ : GeneralizedHeytingAlgebra α
      a b c d : α
      ⊢ Iff (And (LE.le a (HImp.himp d c)) (LE.le b (HImp.himp d c))) (And (LE.le d  …
    -/
    simp_rw [le_himp_comm]
    /-
      🎉 no goals
    -/


theorem himp_le_himp_left (h : a ≤ b) : c ⇨ a ≤ c ⇨ b :=
  le_himp_iff.2 <| himp_inf_le.trans h


theorem himp_le_himp_right (h : a ≤ b) : b ⇨ c ≤ a ⇨ c :=
  le_himp_iff.2 <| (inf_le_inf_left _ h).trans himp_inf_le


theorem himp_le_himp (hab : a ≤ b) (hcd : c ≤ d) : b ⇨ c ≤ a ⇨ d :=
  (himp_le_himp_right hab).trans <| himp_le_himp_left hcd


@[simp]
theorem sup_himp_self_left (a b : α) : a ⊔ b ⇨ a = b ⇨ a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    ⊢ Eq (HImp.himp (Max.max a b) a) (HImp.himp b a)
  -/
  rw [sup_himp_distrib, himp_self, top_inf_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem sup_himp_self_right (a b : α) : a ⊔ b ⇨ b = a ⇨ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    ⊢ Eq (HImp.himp (Max.max a b) b) (HImp.himp a b)
  -/
  rw [sup_himp_distrib, himp_self, inf_top_eq]
  /-
    🎉 no goals
  -/


theorem Codisjoint.himp_eq_right (h : Codisjoint a b) : b ⇨ a = a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : Codisjoint a b
    ⊢ Eq (HImp.himp b a) a
  -/
  conv_rhs => rw [← @top_himp _ _ a]
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : Codisjoint a b
    ⊢ Eq (HImp.himp b a) (HImp.himp Top.top a)
  -/
  rw [← h.eq_top, sup_himp_self_left]
  /-
    🎉 no goals
  -/


theorem Codisjoint.himp_eq_left (h : Codisjoint a b) : a ⇨ b = b :=
  h.symm.himp_eq_right


theorem Codisjoint.himp_inf_cancel_right (h : Codisjoint a b) : a ⇨ a ⊓ b = b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : Codisjoint a b
    ⊢ Eq (HImp.himp a (Min.min a b)) b
  -/
  rw [himp_inf_distrib, himp_self, top_inf_eq, h.himp_eq_left]
  /-
    🎉 no goals
  -/


theorem Codisjoint.himp_inf_cancel_left (h : Codisjoint a b) : b ⇨ a ⊓ b = a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b : α
    h : Codisjoint a b
    ⊢ Eq (HImp.himp b (Min.min a b)) a
  -/
  rw [himp_inf_distrib, himp_self, inf_top_eq, h.himp_eq_right]
  /-
    🎉 no goals
  -/


/-- See `himp_le` for a stronger version in Boolean algebras. -/
theorem Codisjoint.himp_le_of_right_le (hac : Codisjoint a c) (hba : b ≤ a) : c ⇨ b ≤ a :=
  (himp_le_himp_left hba).trans_eq hac.himp_eq_right


theorem le_himp_himp : a ≤ (a ⇨ b) ⇨ b :=
  le_himp_iff.2 inf_himp_le


                                                             /-
                                                               α : Type u_2
                                                               inst✝ : GeneralizedHeytingAlgebra α
                                                               a b : α
                                                               ⊢ Iff (Eq (HImp.himp b a) (HImp.himp a b)) (Eq a b)
                                                             -/
@[simp] lemma himp_eq_himp_iff : b ⇨ a = a ⇨ b ↔ a = b := by simp [le_antisymm_iff]
                                                             /-
                                                               🎉 no goals
                                                             -/

lemma himp_ne_himp_iff : b ⇨ a ≠ a ⇨ b ↔ a ≠ b := himp_eq_himp_iff.not


theorem himp_triangle (a b c : α) : (a ⇨ b) ⊓ (b ⇨ c) ≤ a ⇨ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ LE.le (Min.min (HImp.himp a b) (HImp.himp b c)) (HImp.himp a c)
  -/
  rw [le_himp_iff, inf_right_comm, ← le_himp_iff]
  /-
    α : Type u_2
    inst✝ : GeneralizedHeytingAlgebra α
    a b c : α
    ⊢ LE.le (Min.min (HImp.himp a b) a) (HImp.himp (HImp.himp b c) c)
  -/
  exact himp_inf_le.trans le_himp_himp
  /-
    🎉 no goals
  -/


theorem himp_inf_himp_cancel (hba : b ≤ a) (hcb : c ≤ b) : (a ⇨ b) ⊓ (b ⇨ c) = a ⇨ c :=
  (himp_triangle _ _ _).antisymm <| le_inf (himp_le_himp_left hcb) (himp_le_himp_right hba)

-- See note [lower instance priority]

instance (priority := 100) GeneralizedHeytingAlgebra.toDistribLattice : DistribLattice α :=
  DistribLattice.ofInfSupLe fun a b c => by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      inst✝ : GeneralizedHeytingAlgebra α
      a✝ b✝ c✝ d a b c : α
      ⊢ LE.le (Min.min a (Max.max b c)) (Max.max (Min.min a b) (Min.min a c))
    -/
    simp_rw [inf_comm a, ← le_himp_iff, sup_le_iff, le_himp_iff, ← sup_le_iff]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance OrderDual.instGeneralizedCoheytingAlgebra : GeneralizedCoheytingAlgebra αᵒᵈ where
  sdiff a b := toDual (ofDual b ⇨ ofDual a)
                           /-
                             ι : Type u_1
                             α : Type u_2
                             β : Type u_3
                             inst✝ : GeneralizedHeytingAlgebra α
                             a✝ b✝ c✝ d : α
                             a b c : OrderDual α
                             ⊢ Iff (LE.le (SDiff.sdiff a b) c) (LE.le a (Max.max b c))
                           -/
  sdiff_le_iff a b c := by rw [sup_comm]; exact le_himp_iff
                                          /-
                                            🎉 no goals
                                          -/


instance Prod.instGeneralizedHeytingAlgebra [GeneralizedHeytingAlgebra β] :
    GeneralizedHeytingAlgebra (α × β) where
  le_himp_iff _ _ _ := and_congr le_himp_iff le_himp_iff


instance Pi.instGeneralizedHeytingAlgebra {α : ι → Type*} [∀ i, GeneralizedHeytingAlgebra (α i)] :
    GeneralizedHeytingAlgebra (∀ i, α i) where
                      /-
                        ι : Type u_1
                        α✝ : Type u_2
                        β : Type u_3
                        inst✝¹ : GeneralizedHeytingAlgebra α✝
                        a b c d : α✝
                        α : ι → Type u_4
                        inst✝ : (i : ι) → GeneralizedHeytingAlgebra (α i)
                        i : (i : ι) → α i
                        ⊢ ∀ (b c : (i : ι) → α i), Iff (LE.le i (HImp.himp b c)) (LE.le (Min.min i b) c)
                      -/
  le_himp_iff i := by simp [le_def]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem sdiff_le_iff : a \ b ≤ c ↔ a ≤ b ⊔ c :=
  GeneralizedCoheytingAlgebra.sdiff_le_iff _ _ _


                                                    /-
                                                      α : Type u_2
                                                      inst✝ : GeneralizedCoheytingAlgebra α
                                                      a b c : α
                                                      ⊢ Iff (LE.le (SDiff.sdiff a b) c) (LE.le a (Max.max c b))
                                                    -/
theorem sdiff_le_iff' : a \ b ≤ c ↔ a ≤ c ⊔ b := by rw [sdiff_le_iff, sup_comm]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                    /-
                                                      α : Type u_2
                                                      inst✝ : GeneralizedCoheytingAlgebra α
                                                      a b c : α
                                                      ⊢ Iff (LE.le (SDiff.sdiff a b) c) (LE.le (SDiff.sdiff a c) b)
                                                    -/
theorem sdiff_le_comm : a \ b ≤ c ↔ a \ c ≤ b := by rw [sdiff_le_iff, sdiff_le_iff']
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem sdiff_le : a \ b ≤ a :=
  sdiff_le_iff.2 le_sup_right


theorem Disjoint.disjoint_sdiff_left (h : Disjoint a b) : Disjoint (a \ c) b :=
  h.mono_left sdiff_le


theorem Disjoint.disjoint_sdiff_right (h : Disjoint a b) : Disjoint a (b \ c) :=
  h.mono_right sdiff_le


                                                    /-
                                                      α : Type u_2
                                                      inst✝ : GeneralizedCoheytingAlgebra α
                                                      a b : α
                                                      ⊢ Iff (LE.le (SDiff.sdiff a b) b) (LE.le a b)
                                                    -/
theorem sdiff_le_iff_left : a \ b ≤ b ↔ a ≤ b := by rw [sdiff_le_iff, sup_idem]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem sdiff_self : a \ a = ⊥ :=
  le_bot_iff.1 <| sdiff_le_iff.2 le_sup_left


theorem le_sup_sdiff : a ≤ b ⊔ a \ b :=
  sdiff_le_iff.1 le_rfl


                                           /-
                                             α : Type u_2
                                             inst✝ : GeneralizedCoheytingAlgebra α
                                             a b : α
                                             ⊢ LE.le a (Max.max (SDiff.sdiff a b) b)
                                           -/
theorem le_sdiff_sup : a ≤ a \ b ⊔ b := by rw [sup_comm, ← sdiff_le_iff]
                                           /-
                                             🎉 no goals
                                           -/


theorem sup_sdiff_left : a ⊔ a \ b = a :=
  sup_of_le_left sdiff_le


theorem sup_sdiff_right : a \ b ⊔ a = a :=
  sup_of_le_right sdiff_le


theorem inf_sdiff_left : a \ b ⊓ a = a \ b :=
  inf_of_le_left sdiff_le


theorem inf_sdiff_right : a ⊓ a \ b = a \ b :=
  inf_of_le_right sdiff_le


@[simp]
theorem sup_sdiff_self (a b : α) : a ⊔ b \ a = a ⊔ b :=
  le_antisymm (sup_le_sup_left sdiff_le _) (sup_le le_sup_left le_sup_sdiff)


@[simp]
                                                           /-
                                                             α : Type u_2
                                                             inst✝ : GeneralizedCoheytingAlgebra α
                                                             a b : α
                                                             ⊢ Eq (Max.max (SDiff.sdiff b a) a) (Max.max b a)
                                                           -/
theorem sdiff_sup_self (a b : α) : b \ a ⊔ a = b ⊔ a := by rw [sup_comm, sup_sdiff_self, sup_comm]
                                                           /-
                                                             🎉 no goals
                                                           -/


alias sup_sdiff_self_left := sdiff_sup_self


alias sup_sdiff_self_right := sup_sdiff_self


theorem sup_sdiff_eq_sup (h : c ≤ a) : a ⊔ b \ c = a ⊔ b :=
  sup_congr_left (sdiff_le.trans le_sup_right) <| le_sup_sdiff.trans <| sup_le_sup_right h _

-- cf. `Set.union_diff_cancel'`

theorem sup_sdiff_cancel' (hab : a ≤ b) (hbc : b ≤ c) : b ⊔ c \ a = c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    hab : LE.le a b
    hbc : LE.le b c
    ⊢ Eq (Max.max b (SDiff.sdiff c a)) c
  -/
  rw [sup_sdiff_eq_sup hab, sup_of_le_right hbc]
  /-
    🎉 no goals
  -/


theorem sup_sdiff_cancel_right (h : a ≤ b) : a ⊔ b \ a = b :=
  sup_sdiff_cancel' le_rfl h


                                                           /-
                                                             α : Type u_2
                                                             inst✝ : GeneralizedCoheytingAlgebra α
                                                             a b : α
                                                             h : LE.le b a
                                                             ⊢ Eq (Max.max (SDiff.sdiff a b) b) a
                                                           -/
theorem sdiff_sup_cancel (h : b ≤ a) : a \ b ⊔ b = a := by rw [sup_comm, sup_sdiff_cancel_right h]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem sup_le_of_le_sdiff_left (h : b ≤ c \ a) (hac : a ≤ c) : a ⊔ b ≤ c :=
  sup_le hac <| h.trans sdiff_le


theorem sup_le_of_le_sdiff_right (h : a ≤ c \ b) (hbc : b ≤ c) : a ⊔ b ≤ c :=
  sup_le (h.trans sdiff_le) hbc


@[simp]
                                                   /-
                                                     α : Type u_2
                                                     inst✝ : GeneralizedCoheytingAlgebra α
                                                     a b : α
                                                     ⊢ Iff (Eq (SDiff.sdiff a b) Bot.bot) (LE.le a b)
                                                   -/
theorem sdiff_eq_bot_iff : a \ b = ⊥ ↔ a ≤ b := by rw [← le_bot_iff, sdiff_le_iff, sup_bot_eq]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem sdiff_bot : a \ ⊥ = a :=
                                  /-
                                    α : Type u_2
                                    inst✝ : GeneralizedCoheytingAlgebra α
                                    a b : α
                                    ⊢ Iff (LE.le (SDiff.sdiff a Bot.bot) b) (LE.le a b)
                                  -/
  eq_of_forall_ge_iff fun b => by rw [sdiff_le_iff, bot_sup_eq]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem bot_sdiff : ⊥ \ a = ⊥ :=
  sdiff_eq_bot_iff.2 bot_le


theorem sdiff_sdiff_sdiff_le_sdiff : (a \ b) \ (a \ c) ≤ c \ b := by
  rw [sdiff_le_iff, sdiff_le_iff, sup_left_comm, sup_sdiff_self, sup_left_comm, sdiff_sup_self,
    sup_left_comm]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ LE.le a (Max.max a (Max.max b c))
  -/
  exact le_sup_left
  /-
    🎉 no goals
  -/


@[simp]
theorem le_sup_sdiff_sup_sdiff : a ≤ b ⊔ (a \ c ⊔ c \ b) := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ LE.le a (Max.max b (Max.max (SDiff.sdiff a c) (SDiff.sdiff c b)))
  -/
  simpa using @sdiff_sdiff_sdiff_le_sdiff
  /-
    🎉 no goals
  -/


theorem sdiff_sdiff (a b c : α) : (a \ b) \ c = a \ (b ⊔ c) :=
                                  /-
                                    α : Type u_2
                                    inst✝ : GeneralizedCoheytingAlgebra α
                                    a b c d : α
                                    ⊢ Iff (LE.le (SDiff.sdiff (SDiff.sdiff a b) c) d) (LE.le (SDiff.sdiff a (Max.m …
                                  -/
  eq_of_forall_ge_iff fun d => by simp_rw [sdiff_le_iff, sup_assoc]
                                  /-
                                    🎉 no goals
                                  -/


theorem sdiff_sdiff_left : (a \ b) \ c = a \ (b ⊔ c) :=
  sdiff_sdiff _ _ _


theorem sdiff_right_comm (a b c : α) : (a \ b) \ c = (a \ c) \ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ Eq (SDiff.sdiff (SDiff.sdiff a b) c) (SDiff.sdiff (SDiff.sdiff a c) b)
  -/
  simp_rw [sdiff_sdiff, sup_comm]
  /-
    🎉 no goals
  -/


theorem sdiff_sdiff_comm : (a \ b) \ c = (a \ c) \ b :=
  sdiff_right_comm _ _ _


@[simp]
                                               /-
                                                 α : Type u_2
                                                 inst✝ : GeneralizedCoheytingAlgebra α
                                                 a b : α
                                                 ⊢ Eq (SDiff.sdiff (SDiff.sdiff a b) b) (SDiff.sdiff a b)
                                               -/
theorem sdiff_idem : (a \ b) \ b = a \ b := by rw [sdiff_sdiff_left, sup_idem]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
                                                 /-
                                                   α : Type u_2
                                                   inst✝ : GeneralizedCoheytingAlgebra α
                                                   a b : α
                                                   ⊢ Eq (SDiff.sdiff (SDiff.sdiff a b) a) Bot.bot
                                                 -/
theorem sdiff_sdiff_self : (a \ b) \ a = ⊥ := by rw [sdiff_sdiff_comm, sdiff_self, bot_sdiff]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem sup_sdiff_distrib (a b c : α) : (a ⊔ b) \ c = a \ c ⊔ b \ c :=
                                  /-
                                    α : Type u_2
                                    inst✝ : GeneralizedCoheytingAlgebra α
                                    a b c d : α
                                    ⊢ Iff (LE.le (SDiff.sdiff (Max.max a b) c) d) (LE.le (Max.max (SDiff.sdiff a c …
                                  -/
  eq_of_forall_ge_iff fun d => by simp_rw [sdiff_le_iff, sup_le_iff, sdiff_le_iff]
                                  /-
                                    🎉 no goals
                                  -/


theorem sdiff_inf_distrib (a b c : α) : a \ (b ⊓ c) = a \ b ⊔ a \ c :=
  eq_of_forall_ge_iff fun d => by
    /-
      α : Type u_2
      inst✝ : GeneralizedCoheytingAlgebra α
      a b c d : α
      ⊢ Iff (LE.le (SDiff.sdiff a (Min.min b c)) d) (LE.le (Max.max (SDiff.sdiff a b …
    -/
    rw [sup_le_iff, sdiff_le_comm, le_inf_iff]
    /-
      α : Type u_2
      inst✝ : GeneralizedCoheytingAlgebra α
      a b c d : α
      ⊢ Iff (And (LE.le (SDiff.sdiff a d) b) (LE.le (SDiff.sdiff a d) c)) (And (LE.l …
    -/
    simp_rw [sdiff_le_comm]
    /-
      🎉 no goals
    -/


theorem sup_sdiff : (a ⊔ b) \ c = a \ c ⊔ b \ c :=
  sup_sdiff_distrib _ _ _


@[simp]
                                                         /-
                                                           α : Type u_2
                                                           inst✝ : GeneralizedCoheytingAlgebra α
                                                           a b : α
                                                           ⊢ Eq (SDiff.sdiff (Max.max a b) b) (SDiff.sdiff a b)
                                                         -/
theorem sup_sdiff_right_self : (a ⊔ b) \ b = a \ b := by rw [sup_sdiff, sdiff_self, sup_bot_eq]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                        /-
                                                          α : Type u_2
                                                          inst✝ : GeneralizedCoheytingAlgebra α
                                                          a b : α
                                                          ⊢ Eq (SDiff.sdiff (Max.max a b) a) (SDiff.sdiff b a)
                                                        -/
theorem sup_sdiff_left_self : (a ⊔ b) \ a = b \ a := by rw [sup_comm, sup_sdiff_right_self]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[gcongr]
theorem sdiff_le_sdiff_right (h : a ≤ b) : a \ c ≤ b \ c :=
  sdiff_le_iff.2 <| h.trans <| le_sup_sdiff


@[gcongr]
theorem sdiff_le_sdiff_left (h : a ≤ b) : c \ b ≤ c \ a :=
  sdiff_le_iff.2 <| le_sup_sdiff.trans <| sup_le_sup_right h _


@[gcongr]
theorem sdiff_le_sdiff (hab : a ≤ b) (hcd : c ≤ d) : a \ d ≤ b \ c :=
  (sdiff_le_sdiff_right hab).trans <| sdiff_le_sdiff_left hcd

-- cf. `IsCompl.inf_sup`

theorem sdiff_inf : a \ (b ⊓ c) = a \ b ⊔ a \ c :=
  sdiff_inf_distrib _ _ _


@[simp]
theorem sdiff_inf_self_left (a b : α) : a \ (a ⊓ b) = a \ b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (SDiff.sdiff a (Min.min a b)) (SDiff.sdiff a b)
  -/
  rw [sdiff_inf, sdiff_self, bot_sup_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem sdiff_inf_self_right (a b : α) : b \ (a ⊓ b) = b \ a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    ⊢ Eq (SDiff.sdiff b (Min.min a b)) (SDiff.sdiff b a)
  -/
  rw [sdiff_inf, sdiff_self, sup_bot_eq]
  /-
    🎉 no goals
  -/


theorem Disjoint.sdiff_eq_left (h : Disjoint a b) : a \ b = a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : Disjoint a b
    ⊢ Eq (SDiff.sdiff a b) a
  -/
  conv_rhs => rw [← @sdiff_bot _ _ a]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : Disjoint a b
    ⊢ Eq (SDiff.sdiff a b) (SDiff.sdiff a Bot.bot)
  -/
  rw [← h.eq_bot, sdiff_inf_self_left]
  /-
    🎉 no goals
  -/


theorem Disjoint.sdiff_eq_right (h : Disjoint a b) : b \ a = b :=
  h.symm.sdiff_eq_left


theorem Disjoint.sup_sdiff_cancel_left (h : Disjoint a b) : (a ⊔ b) \ a = b := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : Disjoint a b
    ⊢ Eq (SDiff.sdiff (Max.max a b) a) b
  -/
  rw [sup_sdiff, sdiff_self, bot_sup_eq, h.sdiff_eq_right]
  /-
    🎉 no goals
  -/


theorem Disjoint.sup_sdiff_cancel_right (h : Disjoint a b) : (a ⊔ b) \ b = a := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b : α
    h : Disjoint a b
    ⊢ Eq (SDiff.sdiff (Max.max a b) b) a
  -/
  rw [sup_sdiff, sdiff_self, sup_bot_eq, h.sdiff_eq_left]
  /-
    🎉 no goals
  -/


/-- See `le_sdiff` for a stronger version in generalised Boolean algebras. -/
theorem Disjoint.le_sdiff_of_le_left (hac : Disjoint a c) (hab : a ≤ b) : a ≤ b \ c :=
  hac.sdiff_eq_left.ge.trans <| sdiff_le_sdiff_right hab


theorem sdiff_sdiff_le : a \ (a \ b) ≤ b :=
  sdiff_le_iff.2 le_sdiff_sup


                                                               /-
                                                                 α : Type u_2
                                                                 inst✝ : GeneralizedCoheytingAlgebra α
                                                                 a b : α
                                                                 ⊢ Iff (Eq (SDiff.sdiff a b) (SDiff.sdiff b a)) (Eq a b)
                                                               -/
@[simp] lemma sdiff_eq_sdiff_iff : a \ b = b \ a ↔ a = b := by simp [le_antisymm_iff]
                                                               /-
                                                                 🎉 no goals
                                                               -/

lemma sdiff_ne_sdiff_iff : a \ b ≠ b \ a ↔ a ≠ b := sdiff_eq_sdiff_iff.not


theorem sdiff_triangle (a b c : α) : a \ c ≤ a \ b ⊔ b \ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ LE.le (SDiff.sdiff a c) (Max.max (SDiff.sdiff a b) (SDiff.sdiff b c))
  -/
  rw [sdiff_le_iff, sup_left_comm, ← sdiff_le_iff]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    ⊢ LE.le (SDiff.sdiff a (SDiff.sdiff a b)) (Max.max c (SDiff.sdiff b c))
  -/
  exact sdiff_sdiff_le.trans le_sup_sdiff
  /-
    🎉 no goals
  -/


theorem sdiff_sup_sdiff_cancel (hba : b ≤ a) (hcb : c ≤ b) : a \ b ⊔ b \ c = a \ c :=
  (sdiff_triangle _ _ _).antisymm' <| sup_le (sdiff_le_sdiff_left hcb) (sdiff_le_sdiff_right hba)


theorem sdiff_le_sdiff_of_sup_le_sup_left (h : c ⊔ a ≤ c ⊔ b) : a \ c ≤ b \ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    h : LE.le (Max.max c a) (Max.max c b)
    ⊢ LE.le (SDiff.sdiff a c) (SDiff.sdiff b c)
  -/
  rw [← sup_sdiff_left_self, ← @sup_sdiff_left_self _ _ _ b]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    h : LE.le (Max.max c a) (Max.max c b)
    ⊢ LE.le (SDiff.sdiff (Max.max c a) c) (SDiff.sdiff (Max.max c b) c)
  -/
  exact sdiff_le_sdiff_right h
  /-
    🎉 no goals
  -/


theorem sdiff_le_sdiff_of_sup_le_sup_right (h : a ⊔ c ≤ b ⊔ c) : a \ c ≤ b \ c := by
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    h : LE.le (Max.max a c) (Max.max b c)
    ⊢ LE.le (SDiff.sdiff a c) (SDiff.sdiff b c)
  -/
  rw [← sup_sdiff_right_self, ← @sup_sdiff_right_self _ _ b]
  /-
    α : Type u_2
    inst✝ : GeneralizedCoheytingAlgebra α
    a b c : α
    h : LE.le (Max.max a c) (Max.max b c)
    ⊢ LE.le (SDiff.sdiff (Max.max a c) c) (SDiff.sdiff (Max.max b c) c)
  -/
  exact sdiff_le_sdiff_right h
  /-
    🎉 no goals
  -/


@[simp]
theorem inf_sdiff_sup_left : a \ c ⊓ (a ⊔ b) = a \ c :=
  inf_of_le_left <| sdiff_le.trans le_sup_left


@[simp]
theorem inf_sdiff_sup_right : a \ c ⊓ (b ⊔ a) = a \ c :=
  inf_of_le_left <| sdiff_le.trans le_sup_right

-- See note [lower instance priority]

instance (priority := 100) GeneralizedCoheytingAlgebra.toDistribLattice : DistribLattice α :=
  { ‹GeneralizedCoheytingAlgebra α› with
    le_sup_inf :=
                      /-
                        ι : Type u_1
                        α : Type u_2
                        β : Type u_3
                        inst✝ : GeneralizedCoheytingAlgebra α
                        a✝ b✝ c✝ d a b c : α
                        ⊢ LE.le (Min.min (Max.max a b) (Max.max a c)) (Max.max a (Min.min b c))
                      -/
      fun a b c => by simp_rw [← sdiff_le_iff, le_inf_iff, sdiff_le_iff, ← le_inf_iff]; rfl }
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


instance OrderDual.instGeneralizedHeytingAlgebra : GeneralizedHeytingAlgebra αᵒᵈ where
  himp := fun a b => toDual (ofDual b \ ofDual a)
                                 /-
                                   ι : Type u_1
                                   α : Type u_2
                                   β : Type u_3
                                   inst✝ : GeneralizedCoheytingAlgebra α
                                   a✝ b✝ c✝ d : α
                                   a b c : OrderDual α
                                   ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le (Min.min a b) c)
                                 -/
  le_himp_iff := fun a b c => by rw [inf_comm]; exact sdiff_le_iff
                                                /-
                                                  🎉 no goals
                                                -/


instance Prod.instGeneralizedCoheytingAlgebra [GeneralizedCoheytingAlgebra β] :
    GeneralizedCoheytingAlgebra (α × β) where
  sdiff_le_iff _ _ _ := and_congr sdiff_le_iff sdiff_le_iff


instance Pi.instGeneralizedCoheytingAlgebra {α : ι → Type*}
    [∀ i, GeneralizedCoheytingAlgebra (α i)] : GeneralizedCoheytingAlgebra (∀ i, α i) where
                       /-
                         ι : Type u_1
                         α✝ : Type u_2
                         β : Type u_3
                         inst✝¹ : GeneralizedCoheytingAlgebra α✝
                         a b c d : α✝
                         α : ι → Type u_4
                         inst✝ : (i : ι) → GeneralizedCoheytingAlgebra (α i)
                         i : (i : ι) → α i
                         ⊢ ∀ (b c : (i : ι) → α i), Iff (LE.le (SDiff.sdiff i b) c) (LE.le i (Max.max b …
                       -/
  sdiff_le_iff i := by simp [le_def]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem himp_bot (a : α) : a ⇨ ⊥ = aᶜ :=
  HeytingAlgebra.himp_bot _


@[simp]
theorem bot_himp (a : α) : ⊥ ⇨ a = ⊤ :=
  himp_eq_top_iff.2 bot_le


theorem compl_sup_distrib (a b : α) : (a ⊔ b)ᶜ = aᶜ ⊓ bᶜ := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Eq (HasCompl.compl (Max.max a b)) (Min.min (HasCompl.compl a) (HasCompl.comp …
  -/
  simp_rw [← himp_bot, sup_himp_distrib]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_sup : (a ⊔ b)ᶜ = aᶜ ⊓ bᶜ :=
  compl_sup_distrib _ _


theorem compl_le_himp : aᶜ ≤ a ⇨ b :=
  (himp_bot _).ge.trans <| himp_le_himp_left bot_le


theorem compl_sup_le_himp : aᶜ ⊔ b ≤ a ⇨ b :=
  sup_le compl_le_himp le_himp


theorem sup_compl_le_himp : b ⊔ aᶜ ≤ a ⇨ b :=
  sup_le le_himp compl_le_himp

-- `p → ¬ p ↔ ¬ p`

@[simp]
                                               /-
                                                 α : Type u_2
                                                 inst✝ : HeytingAlgebra α
                                                 a : α
                                                 ⊢ Eq (HImp.himp a (HasCompl.compl a)) (HasCompl.compl a)
                                               -/
theorem himp_compl (a : α) : a ⇨ aᶜ = aᶜ := by rw [← himp_bot, himp_himp, inf_idem]
                                               /-
                                                 🎉 no goals
                                               -/

-- `p → ¬ q ↔ q → ¬ p`

                                                          /-
                                                            α : Type u_2
                                                            inst✝ : HeytingAlgebra α
                                                            a b : α
                                                            ⊢ Eq (HImp.himp a (HasCompl.compl b)) (HImp.himp b (HasCompl.compl a))
                                                          -/
theorem himp_compl_comm (a b : α) : a ⇨ bᶜ = b ⇨ aᶜ := by simp_rw [← himp_bot, himp_left_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem le_compl_iff_disjoint_right : a ≤ bᶜ ↔ Disjoint a b := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Iff (LE.le a (HasCompl.compl b)) (Disjoint a b)
  -/
  rw [← himp_bot, le_himp_iff, disjoint_iff_inf_le]
  /-
    🎉 no goals
  -/


theorem le_compl_iff_disjoint_left : a ≤ bᶜ ↔ Disjoint b a :=
  le_compl_iff_disjoint_right.trans disjoint_comm


theorem le_compl_comm : a ≤ bᶜ ↔ b ≤ aᶜ := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Iff (LE.le a (HasCompl.compl b)) (LE.le b (HasCompl.compl a))
  -/
  rw [le_compl_iff_disjoint_right, le_compl_iff_disjoint_left]
  /-
    🎉 no goals
  -/


alias ⟨_, Disjoint.le_compl_right⟩ := le_compl_iff_disjoint_right


alias ⟨_, Disjoint.le_compl_left⟩ := le_compl_iff_disjoint_left


alias le_compl_iff_le_compl := le_compl_comm


alias ⟨le_compl_of_le_compl, _⟩ := le_compl_comm


theorem disjoint_compl_left : Disjoint aᶜ a :=
  disjoint_iff_inf_le.mpr <| le_himp_iff.1 (himp_bot _).ge


theorem disjoint_compl_right : Disjoint a aᶜ :=
  disjoint_compl_left.symm


theorem LE.le.disjoint_compl_left (h : b ≤ a) : Disjoint aᶜ b :=
  _root_.disjoint_compl_left.mono_right h


theorem LE.le.disjoint_compl_right (h : a ≤ b) : Disjoint a bᶜ :=
  _root_.disjoint_compl_right.mono_left h


theorem IsCompl.compl_eq (h : IsCompl a b) : aᶜ = b :=
  h.1.le_compl_left.antisymm' <| Disjoint.le_of_codisjoint disjoint_compl_left h.2


theorem IsCompl.eq_compl (h : IsCompl a b) : a = bᶜ :=
  h.1.le_compl_right.antisymm <| Disjoint.le_of_codisjoint disjoint_compl_left h.2.symm


theorem compl_unique (h₀ : a ⊓ b = ⊥) (h₁ : a ⊔ b = ⊤) : aᶜ = b :=
  (IsCompl.of_eq h₀ h₁).compl_eq


@[simp]
theorem inf_compl_self (a : α) : a ⊓ aᶜ = ⊥ :=
  disjoint_compl_right.eq_bot


@[simp]
theorem compl_inf_self (a : α) : aᶜ ⊓ a = ⊥ :=
  disjoint_compl_left.eq_bot


theorem inf_compl_eq_bot : a ⊓ aᶜ = ⊥ :=
  inf_compl_self _


theorem compl_inf_eq_bot : aᶜ ⊓ a = ⊥ :=
  compl_inf_self _


@[simp]
theorem compl_top : (⊤ : α)ᶜ = ⊥ :=
                                  /-
                                    α : Type u_2
                                    inst✝ : HeytingAlgebra α
                                    a : α
                                    ⊢ Iff (LE.le a (HasCompl.compl Top.top)) (LE.le a Bot.bot)
                                  -/
  eq_of_forall_le_iff fun a => by rw [le_compl_iff_disjoint_right, disjoint_top, le_bot_iff]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
                                       /-
                                         α : Type u_2
                                         inst✝ : HeytingAlgebra α
                                         ⊢ Eq (HasCompl.compl Bot.bot) Top.top
                                       -/
theorem compl_bot : (⊥ : α)ᶜ = ⊤ := by rw [← himp_bot, himp_self]
                                       /-
                                         🎉 no goals
                                       -/


@[simp] theorem le_compl_self : a ≤ aᶜ ↔ a = ⊥ := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a : α
    ⊢ Iff (LE.le a (HasCompl.compl a)) (Eq a Bot.bot)
  -/
  rw [le_compl_iff_disjoint_left, disjoint_self]
  /-
    🎉 no goals
  -/


@[simp] theorem ne_compl_self [Nontrivial α] : a ≠ aᶜ := by
  /-
    α : Type u_2
    inst✝¹ : HeytingAlgebra α
    a : α
    inst✝ : Nontrivial α
    ⊢ Ne a (HasCompl.compl a)
  -/
  intro h
  /-
    α : Type u_2
    inst✝¹ : HeytingAlgebra α
    a : α
    inst✝ : Nontrivial α
    h : Eq a (HasCompl.compl a)
    ⊢ False
  -/
  cases le_compl_self.1 (le_of_eq h)
  /-
    case refl
    α : Type u_2
    inst✝¹ : HeytingAlgebra α
    inst✝ : Nontrivial α
    h : Eq Bot.bot (HasCompl.compl Bot.bot)
    ⊢ False
  -/
  simp at h
  /-
    🎉 no goals
  -/


@[simp] theorem compl_ne_self [Nontrivial α] : aᶜ ≠ a :=
  ne_comm.1 ne_compl_self


@[simp] theorem lt_compl_self [Nontrivial α] : a < aᶜ ↔ a = ⊥ := by
  /-
    α : Type u_2
    inst✝¹ : HeytingAlgebra α
    a : α
    inst✝ : Nontrivial α
    ⊢ Iff (LT.lt a (HasCompl.compl a)) (Eq a Bot.bot)
  -/
  rw [lt_iff_le_and_ne]; simp
                         /-
                           🎉 no goals
                         -/


theorem le_compl_compl : a ≤ aᶜᶜ :=
  disjoint_compl_right.le_compl_right


theorem compl_anti : Antitone (compl : α → α) := fun _ _ h =>
  le_compl_comm.1 <| h.trans le_compl_compl


@[gcongr]
theorem compl_le_compl (h : a ≤ b) : bᶜ ≤ aᶜ :=
  compl_anti h


@[simp]
theorem compl_compl_compl (a : α) : aᶜᶜᶜ = aᶜ :=
  (compl_anti le_compl_compl).antisymm le_compl_compl


@[simp]
theorem disjoint_compl_compl_left_iff : Disjoint aᶜᶜ b ↔ Disjoint a b := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Iff (Disjoint (HasCompl.compl (HasCompl.compl a)) b) (Disjoint a b)
  -/
  simp_rw [← le_compl_iff_disjoint_left, compl_compl_compl]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_compl_compl_right_iff : Disjoint a bᶜᶜ ↔ Disjoint a b := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Iff (Disjoint a (HasCompl.compl (HasCompl.compl b))) (Disjoint a b)
  -/
  simp_rw [← le_compl_iff_disjoint_right, compl_compl_compl]
  /-
    🎉 no goals
  -/


theorem compl_sup_compl_le : aᶜ ⊔ bᶜ ≤ (a ⊓ b)ᶜ :=
  sup_le (compl_anti inf_le_left) <| compl_anti inf_le_right


theorem compl_compl_inf_distrib (a b : α) : (a ⊓ b)ᶜᶜ = aᶜᶜ ⊓ bᶜᶜ := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Eq (HasCompl.compl (HasCompl.compl (Min.min a b))) (Min.min (HasCompl.compl  …
  -/
  refine ((compl_anti compl_sup_compl_le).trans (compl_sup_distrib _ _).le).antisymm ?_
  rw [le_compl_iff_disjoint_right, disjoint_assoc, disjoint_compl_compl_left_iff,
    disjoint_left_comm, disjoint_compl_compl_left_iff, ← disjoint_assoc, inf_comm]
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Disjoint (Min.min a b) (HasCompl.compl (Min.min a b))
  -/
  exact disjoint_compl_right
  /-
    🎉 no goals
  -/


theorem compl_compl_himp_distrib (a b : α) : (a ⇨ b)ᶜᶜ = aᶜᶜ ⇨ bᶜᶜ := by
  /-
    α : Type u_2
    inst✝ : HeytingAlgebra α
    a b : α
    ⊢ Eq (HasCompl.compl (HasCompl.compl (HImp.himp a b))) (HImp.himp (HasCompl.co …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_2
      inst✝ : HeytingAlgebra α
      a b : α
      ⊢ LE.le (HasCompl.compl (HasCompl.compl (HImp.himp a b))) (HImp.himp (HasCompl …
    -/
  · rw [le_himp_iff, ← compl_compl_inf_distrib]
    /-
      case a
      α : Type u_2
      inst✝ : HeytingAlgebra α
      a b : α
      ⊢ LE.le (HasCompl.compl (HasCompl.compl (Min.min (HImp.himp a b) a))) (HasComp …
    -/
    exact compl_anti (compl_anti himp_inf_le)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_2
      inst✝ : HeytingAlgebra α
      a b : α
      ⊢ LE.le (HImp.himp (HasCompl.compl (HasCompl.compl a)) (HasCompl.compl (HasCom …
    -/
  · refine le_compl_comm.1 ((compl_anti compl_sup_le_himp).trans ?_)
    rw [compl_sup_distrib, le_compl_iff_disjoint_right, disjoint_right_comm, ←
      le_compl_iff_disjoint_right]
    /-
      case a
      α : Type u_2
      inst✝ : HeytingAlgebra α
      a b : α
      ⊢ LE.le (Min.min (HasCompl.compl (HasCompl.compl a)) (HImp.himp (HasCompl.comp …
    -/
    exact inf_himp_le
    /-
      🎉 no goals
    -/


instance OrderDual.instCoheytingAlgebra : CoheytingAlgebra αᵒᵈ where
  hnot := toDual ∘ compl ∘ ofDual
  sdiff a b := toDual (ofDual b ⇨ ofDual a)
                           /-
                             ι : Type u_1
                             α : Type u_2
                             β : Type u_3
                             inst✝ : HeytingAlgebra α
                             a✝ b✝ : α
                             a b c : OrderDual α
                             ⊢ Iff (LE.le (SDiff.sdiff a b) c) (LE.le a (Max.max b c))
                           -/
  sdiff_le_iff a b c := by rw [sup_comm]; exact le_himp_iff
                                          /-
                                            🎉 no goals
                                          -/
  top_sdiff := @himp_bot α _


@[simp]
theorem ofDual_hnot (a : αᵒᵈ) : ofDual (￢a) = (ofDual a)ᶜ :=
  rfl


@[simp]
theorem toDual_compl (a : α) : toDual aᶜ = ￢toDual a :=
  rfl


instance Prod.instHeytingAlgebra [HeytingAlgebra β] : HeytingAlgebra (α × β) where
    himp_bot a := Prod.ext_iff.2 ⟨himp_bot a.1, himp_bot a.2⟩


instance Pi.instHeytingAlgebra {α : ι → Type*} [∀ i, HeytingAlgebra (α i)] :
    HeytingAlgebra (∀ i, α i) where
  himp_bot f := funext fun i ↦ himp_bot (f i)


@[simp]
theorem top_sdiff' (a : α) : ⊤ \ a = ￢a :=
  CoheytingAlgebra.top_sdiff _


@[simp]
theorem sdiff_top (a : α) : a \ ⊤ = ⊥ :=
  sdiff_eq_bot_iff.2 le_top


theorem hnot_inf_distrib (a b : α) : ￢(a ⊓ b) = ￢a ⊔ ￢b := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Eq (HNot.hnot (Min.min a b)) (Max.max (HNot.hnot a) (HNot.hnot b))
  -/
  simp_rw [← top_sdiff', sdiff_inf_distrib]
  /-
    🎉 no goals
  -/


theorem sdiff_le_hnot : a \ b ≤ ￢b :=
  (sdiff_le_sdiff_right le_top).trans_eq <| top_sdiff' _


theorem sdiff_le_inf_hnot : a \ b ≤ a ⊓ ￢b :=
  le_inf sdiff_le sdiff_le_hnot

-- See note [lower instance priority]

instance (priority := 100) CoheytingAlgebra.toDistribLattice : DistribLattice α :=
  { ‹CoheytingAlgebra α› with
    le_sup_inf :=
                      /-
                        ι : Type u_1
                        α : Type u_2
                        β : Type u_3
                        inst✝ : CoheytingAlgebra α
                        a✝ b✝ a b c : α
                        ⊢ LE.le (Min.min (Max.max a b) (Max.max a c)) (Max.max a (Min.min b c))
                      -/
      fun a b c => by simp_rw [← sdiff_le_iff, le_inf_iff, sdiff_le_iff, ← le_inf_iff]; rfl }
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[simp]
                                               /-
                                                 α : Type u_2
                                                 inst✝ : CoheytingAlgebra α
                                                 a : α
                                                 ⊢ Eq (SDiff.sdiff (HNot.hnot a) a) (HNot.hnot a)
                                               -/
theorem hnot_sdiff (a : α) : ￢a \ a = ￢a := by rw [← top_sdiff', sdiff_sdiff, sup_idem]
                                               /-
                                                 🎉 no goals
                                               -/


                                                          /-
                                                            α : Type u_2
                                                            inst✝ : CoheytingAlgebra α
                                                            a b : α
                                                            ⊢ Eq (SDiff.sdiff (HNot.hnot a) b) (SDiff.sdiff (HNot.hnot b) a)
                                                          -/
theorem hnot_sdiff_comm (a b : α) : ￢a \ b = ￢b \ a := by simp_rw [← top_sdiff', sdiff_right_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem hnot_le_iff_codisjoint_right : ￢a ≤ b ↔ Codisjoint a b := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Iff (LE.le (HNot.hnot a) b) (Codisjoint a b)
  -/
  rw [← top_sdiff', sdiff_le_iff, codisjoint_iff_le_sup]
  /-
    🎉 no goals
  -/


theorem hnot_le_iff_codisjoint_left : ￢a ≤ b ↔ Codisjoint b a :=
  hnot_le_iff_codisjoint_right.trans codisjoint_comm


theorem hnot_le_comm : ￢a ≤ b ↔ ￢b ≤ a := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Iff (LE.le (HNot.hnot a) b) (LE.le (HNot.hnot b) a)
  -/
  rw [hnot_le_iff_codisjoint_right, hnot_le_iff_codisjoint_left]
  /-
    🎉 no goals
  -/


alias ⟨_, Codisjoint.hnot_le_right⟩ := hnot_le_iff_codisjoint_right


alias ⟨_, Codisjoint.hnot_le_left⟩ := hnot_le_iff_codisjoint_left


theorem codisjoint_hnot_right : Codisjoint a (￢a) :=
  codisjoint_iff_le_sup.2 <| sdiff_le_iff.1 (top_sdiff' _).le


theorem codisjoint_hnot_left : Codisjoint (￢a) a :=
  codisjoint_hnot_right.symm


theorem LE.le.codisjoint_hnot_left (h : a ≤ b) : Codisjoint (￢a) b :=
  _root_.codisjoint_hnot_left.mono_right h


theorem LE.le.codisjoint_hnot_right (h : b ≤ a) : Codisjoint a (￢b) :=
  _root_.codisjoint_hnot_right.mono_left h


theorem IsCompl.hnot_eq (h : IsCompl a b) : ￢a = b :=
  h.2.hnot_le_right.antisymm <| Disjoint.le_of_codisjoint h.1.symm codisjoint_hnot_right


theorem IsCompl.eq_hnot (h : IsCompl a b) : a = ￢b :=
  h.2.hnot_le_left.antisymm' <| Disjoint.le_of_codisjoint h.1 codisjoint_hnot_right


@[simp]
theorem sup_hnot_self (a : α) : a ⊔ ￢a = ⊤ :=
  Codisjoint.eq_top codisjoint_hnot_right


@[simp]
theorem hnot_sup_self (a : α) : ￢a ⊔ a = ⊤ :=
  Codisjoint.eq_top codisjoint_hnot_left


@[simp]
theorem hnot_bot : ￢(⊥ : α) = ⊤ :=
                                  /-
                                    α : Type u_2
                                    inst✝ : CoheytingAlgebra α
                                    a : α
                                    ⊢ Iff (LE.le (HNot.hnot Bot.bot) a) (LE.le Top.top a)
                                  -/
  eq_of_forall_ge_iff fun a => by rw [hnot_le_iff_codisjoint_left, codisjoint_bot, top_le_iff]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
                                      /-
                                        α : Type u_2
                                        inst✝ : CoheytingAlgebra α
                                        ⊢ Eq (HNot.hnot Top.top) Bot.bot
                                      -/
theorem hnot_top : ￢(⊤ : α) = ⊥ := by rw [← top_sdiff', sdiff_self]
                                      /-
                                        🎉 no goals
                                      -/


theorem hnot_hnot_le : ￢￢a ≤ a :=
  codisjoint_hnot_right.hnot_le_left


theorem hnot_anti : Antitone (hnot : α → α) := fun _ _ h => hnot_le_comm.1 <| hnot_hnot_le.trans h


theorem hnot_le_hnot (h : a ≤ b) : ￢b ≤ ￢a :=
  hnot_anti h


@[simp]
theorem hnot_hnot_hnot (a : α) : ￢￢￢a = ￢a :=
  hnot_hnot_le.antisymm <| hnot_anti hnot_hnot_le


@[simp]
theorem codisjoint_hnot_hnot_left_iff : Codisjoint (￢￢a) b ↔ Codisjoint a b := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Iff (Codisjoint (HNot.hnot (HNot.hnot a)) b) (Codisjoint a b)
  -/
  simp_rw [← hnot_le_iff_codisjoint_right, hnot_hnot_hnot]
  /-
    🎉 no goals
  -/


@[simp]
theorem codisjoint_hnot_hnot_right_iff : Codisjoint a (￢￢b) ↔ Codisjoint a b := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Iff (Codisjoint a (HNot.hnot (HNot.hnot b))) (Codisjoint a b)
  -/
  simp_rw [← hnot_le_iff_codisjoint_left, hnot_hnot_hnot]
  /-
    🎉 no goals
  -/


theorem le_hnot_inf_hnot : ￢(a ⊔ b) ≤ ￢a ⊓ ￢b :=
  le_inf (hnot_anti le_sup_left) <| hnot_anti le_sup_right


theorem hnot_hnot_sup_distrib (a b : α) : ￢￢(a ⊔ b) = ￢￢a ⊔ ￢￢b := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Eq (HNot.hnot (HNot.hnot (Max.max a b))) (Max.max (HNot.hnot (HNot.hnot a))  …
  -/
  refine ((hnot_inf_distrib _ _).ge.trans <| hnot_anti le_hnot_inf_hnot).antisymm' ?_
  rw [hnot_le_iff_codisjoint_left, codisjoint_assoc, codisjoint_hnot_hnot_left_iff,
    codisjoint_left_comm, codisjoint_hnot_hnot_left_iff, ← codisjoint_assoc, sup_comm]
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Codisjoint (Max.max a b) (HNot.hnot (Max.max a b))
  -/
  exact codisjoint_hnot_right
  /-
    🎉 no goals
  -/


theorem hnot_hnot_sdiff_distrib (a b : α) : ￢￢(a \ b) = ￢￢a \ ￢￢b := by
  /-
    α : Type u_2
    inst✝ : CoheytingAlgebra α
    a b : α
    ⊢ Eq (HNot.hnot (HNot.hnot (SDiff.sdiff a b))) (SDiff.sdiff (HNot.hnot (HNot.h …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_2
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (HNot.hnot (HNot.hnot (SDiff.sdiff a b))) (SDiff.sdiff (HNot.hnot (HNo …
    -/
  · refine hnot_le_comm.1 ((hnot_anti sdiff_le_inf_hnot).trans' ?_)
    rw [hnot_inf_distrib, hnot_le_iff_codisjoint_right, codisjoint_left_comm, ←
      hnot_le_iff_codisjoint_right]
    /-
      case a
      α : Type u_2
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (HNot.hnot (HNot.hnot a)) (Max.max (SDiff.sdiff (HNot.hnot (HNot.hnot  …
    -/
    exact le_sdiff_sup
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_2
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (SDiff.sdiff (HNot.hnot (HNot.hnot a)) (HNot.hnot (HNot.hnot b))) (HNo …
    -/
  · rw [sdiff_le_iff, ← hnot_hnot_sup_distrib]
    /-
      case a
      α : Type u_2
      inst✝ : CoheytingAlgebra α
      a b : α
      ⊢ LE.le (HNot.hnot (HNot.hnot a)) (HNot.hnot (HNot.hnot (Max.max b (SDiff.sdif …
    -/
    exact hnot_anti (hnot_anti le_sup_sdiff)
    /-
      🎉 no goals
    -/


instance OrderDual.instHeytingAlgebra : HeytingAlgebra αᵒᵈ where
  compl := toDual ∘ hnot ∘ ofDual
  himp a b := toDual (ofDual b \ ofDual a)
                          /-
                            ι : Type u_1
                            α : Type u_2
                            β : Type u_3
                            inst✝ : CoheytingAlgebra α
                            a✝ b✝ : α
                            a b c : OrderDual α
                            ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le (Min.min a b) c)
                          -/
  le_himp_iff a b c := by rw [inf_comm]; exact sdiff_le_iff
                                         /-
                                           🎉 no goals
                                         -/
  himp_bot := @top_sdiff' α _


@[simp]
theorem ofDual_compl (a : αᵒᵈ) : ofDual aᶜ = ￢ofDual a :=
  rfl


@[simp]
theorem ofDual_himp (a b : αᵒᵈ) : ofDual (a ⇨ b) = ofDual b \ ofDual a :=
  rfl


@[simp]
theorem toDual_hnot (a : α) : toDual (￢a) = (toDual a)ᶜ :=
  rfl


@[simp]
theorem toDual_sdiff (a b : α) : toDual (a \ b) = toDual b ⇨ toDual a :=
  rfl


instance Prod.instCoheytingAlgebra [CoheytingAlgebra β] : CoheytingAlgebra (α × β) where
  sdiff_le_iff _ _ _ := and_congr sdiff_le_iff sdiff_le_iff
  top_sdiff a := Prod.ext_iff.2 ⟨top_sdiff' a.1, top_sdiff' a.2⟩


instance Pi.instCoheytingAlgebra {α : ι → Type*} [∀ i, CoheytingAlgebra (α i)] :
    CoheytingAlgebra (∀ i, α i) where
  top_sdiff f := funext fun i ↦ top_sdiff' (f i)


theorem compl_le_hnot : aᶜ ≤ ￢a :=
  (disjoint_compl_left : Disjoint _ a).le_of_codisjoint codisjoint_hnot_right


/-- Propositions form a Heyting algebra with implication as Heyting implication and negation as
complement. -/
instance Prop.instHeytingAlgebra : HeytingAlgebra Prop :=
  { Prop.instDistribLattice, Prop.instBoundedOrder with
    himp := (· → ·),
    le_himp_iff := fun _ _ _ => and_imp.symm, himp_bot := fun _ => rfl }


@[simp]
theorem himp_iff_imp (p q : Prop) : p ⇨ q ↔ p → q :=
  Iff.rfl


@[simp]
theorem compl_iff_not (p : Prop) : pᶜ ↔ ¬p :=
  Iff.rfl

-- See note [reducible non-instances]

/-- A bounded linear order is a bi-Heyting algebra by setting
* `a ⇨ b = ⊤` if `a ≤ b` and `a ⇨ b = b` otherwise.
* `a \ b = ⊥` if `a ≤ b` and `a \ b = a` otherwise. -/
abbrev LinearOrder.toBiheytingAlgebra [LinearOrder α] [BoundedOrder α] : BiheytingAlgebra α :=
  { LinearOrder.toLattice, ‹BoundedOrder α› with
    himp := fun a b => if a ≤ b then ⊤ else b,
    compl := fun a => if a = ⊥ then ⊤ else ⊥,
    le_himp_iff := fun a b c => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : LinearOrder α
        inst✝ : BoundedOrder α
        a b c : α
        ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le (Min.min a b) c)
      -/
      change _ ≤ ite _ _ _ ↔ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : LinearOrder α
        inst✝ : BoundedOrder α
        a b c : α
        ⊢ Iff (LE.le a (ite (LE.le b c) Top.top c)) (LE.le (Min.min a b) c)
      -/
      split_ifs with h
        /-
          case pos
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : LinearOrder α
          inst✝ : BoundedOrder α
          a b c : α
          h : LE.le b c
          ⊢ Iff (LE.le a Top.top) (LE.le (Min.min a b) c)
        -/
      · exact iff_of_true le_top (inf_le_of_right_le h)
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : LinearOrder α
          inst✝ : BoundedOrder α
          a b c : α
          h : Not (LE.le b c)
          ⊢ Iff (LE.le a c) (LE.le (Min.min a b) c)
        -/
      · rw [inf_le_iff, or_iff_left h],
        /-
          🎉 no goals
        -/
    himp_bot := fun _ => if_congr le_bot_iff rfl rfl, sdiff := fun a b => if a ≤ b then ⊥ else a,
    hnot := fun a => if a = ⊤ then ⊥ else ⊤,
    sdiff_le_iff := fun a b c => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : LinearOrder α
        inst✝ : BoundedOrder α
        a b c : α
        ⊢ Iff (LE.le (SDiff.sdiff a b) c) (LE.le a (Max.max b c))
      -/
      change ite _ _ _ ≤ _ ↔ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : LinearOrder α
        inst✝ : BoundedOrder α
        a b c : α
        ⊢ Iff (LE.le (ite (LE.le a b) Bot.bot a) c) (LE.le a (Max.max b c))
      -/
      split_ifs with h
        /-
          case pos
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : LinearOrder α
          inst✝ : BoundedOrder α
          a b c : α
          h : LE.le a b
          ⊢ Iff (LE.le Bot.bot c) (LE.le a (Max.max b c))
        -/
      · exact iff_of_true bot_le (le_sup_of_le_left h)
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : LinearOrder α
          inst✝ : BoundedOrder α
          a b c : α
          h : Not (LE.le a b)
          ⊢ Iff (LE.le a c) (LE.le a (Max.max b c))
        -/
      · rw [le_sup_iff, or_iff_right h],
        /-
          🎉 no goals
        -/
    top_sdiff := fun _ => if_congr top_le_iff rfl rfl }


instance OrderDual.instBiheytingAlgebra [BiheytingAlgebra α] : BiheytingAlgebra αᵒᵈ where
  __ := instHeytingAlgebra
  __ := instCoheytingAlgebra


instance Prod.instBiheytingAlgebra [BiheytingAlgebra α] [BiheytingAlgebra β] :
    BiheytingAlgebra (α × β) where
  __ := instHeytingAlgebra
  __ := instCoheytingAlgebra


instance Pi.instBiheytingAlgebra {α : ι → Type*} [∀ i, BiheytingAlgebra (α i)] :
    BiheytingAlgebra (∀ i, α i) where
  __ := instHeytingAlgebra
  __ := instCoheytingAlgebra


/-- Pullback a `GeneralizedHeytingAlgebra` along an injection. -/
protected abbrev Function.Injective.generalizedHeytingAlgebra [Max α] [Min α] [Top α]
    [HImp α] [GeneralizedHeytingAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_top : f ⊤ = ⊤) (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b) : GeneralizedHeytingAlgebra α :=
  { __ := hf.lattice f map_sup map_inf
    __ := ‹Top α›
    __ := ‹HImp α›
    le_top := fun a => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Top α
        inst✝¹ : HImp α
        inst✝ : GeneralizedHeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a : α
        ⊢ LE.le a Top.top
      -/
      change f _ ≤ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Top α
        inst✝¹ : HImp α
        inst✝ : GeneralizedHeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a : α
        ⊢ LE.le (f a) (f Top.top)
      -/
      rw [map_top]
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Top α
        inst✝¹ : HImp α
        inst✝ : GeneralizedHeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a : α
        ⊢ LE.le (f a) Top.top
      -/
      exact le_top,
      /-
        🎉 no goals
      -/
    le_himp_iff := fun a b c => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Top α
        inst✝¹ : HImp α
        inst✝ : GeneralizedHeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a b c : α
        ⊢ Iff (LE.le a (HImp.himp b c)) (LE.le (Min.min a b) c)
      -/
      change f _ ≤ _ ↔ f _ ≤ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Top α
        inst✝¹ : HImp α
        inst✝ : GeneralizedHeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a b c : α
        ⊢ Iff (LE.le (f a) (f (HImp.himp b c))) (LE.le (f (Min.min a b)) (f c))
      -/
      rw [map_himp, map_inf, le_himp_iff] }
      /-
        🎉 no goals
      -/

-- See note [reducible non-instances]

/-- Pullback a `GeneralizedCoheytingAlgebra` along an injection. -/
protected abbrev Function.Injective.generalizedCoheytingAlgebra [Max α] [Min α] [Bot α]
    [SDiff α] [GeneralizedCoheytingAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_bot : f ⊥ = ⊥) (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    GeneralizedCoheytingAlgebra α :=
  { __ := hf.lattice f map_sup map_inf
    __ := ‹Bot α›
    __ := ‹SDiff α›
    bot_le := fun a => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Bot α
        inst✝¹ : SDiff α
        inst✝ : GeneralizedCoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_bot : Eq (f Bot.bot) Bot.bot
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a : α
        ⊢ LE.le Bot.bot a
      -/
      change f _ ≤ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Bot α
        inst✝¹ : SDiff α
        inst✝ : GeneralizedCoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_bot : Eq (f Bot.bot) Bot.bot
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a : α
        ⊢ LE.le (f Bot.bot) (f a)
      -/
      rw [map_bot]
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Bot α
        inst✝¹ : SDiff α
        inst✝ : GeneralizedCoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_bot : Eq (f Bot.bot) Bot.bot
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a : α
        ⊢ LE.le Bot.bot (f a)
      -/
      exact bot_le,
      /-
        🎉 no goals
      -/
    sdiff_le_iff := fun a b c => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Bot α
        inst✝¹ : SDiff α
        inst✝ : GeneralizedCoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_bot : Eq (f Bot.bot) Bot.bot
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a b c : α
        ⊢ Iff (LE.le (SDiff.sdiff a b) c) (LE.le a (Max.max b c))
      -/
      change f _ ≤ _ ↔ f _ ≤ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁴ : Max α
        inst✝³ : Min α
        inst✝² : Bot α
        inst✝¹ : SDiff α
        inst✝ : GeneralizedCoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_bot : Eq (f Bot.bot) Bot.bot
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a b c : α
        ⊢ Iff (LE.le (f (SDiff.sdiff a b)) (f c)) (LE.le (f a) (f (Max.max b c)))
      -/
      rw [map_sdiff, map_sup, sdiff_le_iff] }
      /-
        🎉 no goals
      -/

-- See note [reducible non-instances]

/-- Pullback a `HeytingAlgebra` along an injection. -/
protected abbrev Function.Injective.heytingAlgebra [Max α] [Min α] [Top α] [Bot α]
    [HasCompl α] [HImp α] [HeytingAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) (map_compl : ∀ a, f aᶜ = (f a)ᶜ)
    (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b) : HeytingAlgebra α :=
  { __ := hf.generalizedHeytingAlgebra f map_sup map_inf map_top map_himp
    __ := ‹Bot α›
    __ := ‹HasCompl α›
    bot_le := fun a => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : Max α
        inst✝⁵ : Min α
        inst✝⁴ : Top α
        inst✝³ : Bot α
        inst✝² : HasCompl α
        inst✝¹ : HImp α
        inst✝ : HeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_bot : Eq (f Bot.bot) Bot.bot
        map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a : α
        ⊢ LE.le Bot.bot a
      -/
      change f _ ≤ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : Max α
        inst✝⁵ : Min α
        inst✝⁴ : Top α
        inst✝³ : Bot α
        inst✝² : HasCompl α
        inst✝¹ : HImp α
        inst✝ : HeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_bot : Eq (f Bot.bot) Bot.bot
        map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a : α
        ⊢ LE.le (f Bot.bot) (f a)
      -/
      rw [map_bot]
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : Max α
        inst✝⁵ : Min α
        inst✝⁴ : Top α
        inst✝³ : Bot α
        inst✝² : HasCompl α
        inst✝¹ : HImp α
        inst✝ : HeytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_bot : Eq (f Bot.bot) Bot.bot
        map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
        map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
        a : α
        ⊢ LE.le Bot.bot (f a)
      -/
      exact bot_le,
      /-
        🎉 no goals
      -/
                                  /-
                                    ι : Type u_1
                                    α : Type u_2
                                    β : Type u_3
                                    inst✝⁶ : Max α
                                    inst✝⁵ : Min α
                                    inst✝⁴ : Top α
                                    inst✝³ : Bot α
                                    inst✝² : HasCompl α
                                    inst✝¹ : HImp α
                                    inst✝ : HeytingAlgebra β
                                    f : α → β
                                    hf : Function.Injective f
                                    map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                    map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                    map_top : Eq (f Top.top) Top.top
                                    map_bot : Eq (f Bot.bot) Bot.bot
                                    map_compl : ∀ (a : α), Eq (f (HasCompl.compl a)) (HasCompl.compl (f a))
                                    map_himp : ∀ (a b : α), Eq (f (HImp.himp a b)) (HImp.himp (f a) (f b))
                                    a : α
                                    ⊢ Eq (f (HImp.himp a Bot.bot)) (f (HasCompl.compl a))
                                  -/
    himp_bot := fun a => hf <| by rw [map_himp, map_compl, map_bot, himp_bot] }
                                  /-
                                    🎉 no goals
                                  -/

-- See note [reducible non-instances]

/-- Pullback a `CoheytingAlgebra` along an injection. -/
protected abbrev Function.Injective.coheytingAlgebra [Max α] [Min α] [Top α] [Bot α]
    [HNot α] [SDiff α] [CoheytingAlgebra β] (f : α → β) (hf : Injective f)
    (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b) (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b)
    (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥) (map_hnot : ∀ a, f (￢a) = ￢f a)
    (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) : CoheytingAlgebra α :=
  { __ := hf.generalizedCoheytingAlgebra f map_sup map_inf map_bot map_sdiff
    __ := ‹Top α›
    __ := ‹HNot α›
    le_top := fun a => by
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : Max α
        inst✝⁵ : Min α
        inst✝⁴ : Top α
        inst✝³ : Bot α
        inst✝² : HNot α
        inst✝¹ : SDiff α
        inst✝ : CoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_bot : Eq (f Bot.bot) Bot.bot
        map_hnot : ∀ (a : α), Eq (f (HNot.hnot a)) (HNot.hnot (f a))
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a : α
        ⊢ LE.le a Top.top
      -/
      change f _ ≤ _
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : Max α
        inst✝⁵ : Min α
        inst✝⁴ : Top α
        inst✝³ : Bot α
        inst✝² : HNot α
        inst✝¹ : SDiff α
        inst✝ : CoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_bot : Eq (f Bot.bot) Bot.bot
        map_hnot : ∀ (a : α), Eq (f (HNot.hnot a)) (HNot.hnot (f a))
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a : α
        ⊢ LE.le (f a) (f Top.top)
      -/
      rw [map_top]
      /-
        ι : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : Max α
        inst✝⁵ : Min α
        inst✝⁴ : Top α
        inst✝³ : Bot α
        inst✝² : HNot α
        inst✝¹ : SDiff α
        inst✝ : CoheytingAlgebra β
        f : α → β
        hf : Function.Injective f
        map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
        map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
        map_top : Eq (f Top.top) Top.top
        map_bot : Eq (f Bot.bot) Bot.bot
        map_hnot : ∀ (a : α), Eq (f (HNot.hnot a)) (HNot.hnot (f a))
        map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
        a : α
        ⊢ LE.le (f a) Top.top
      -/
      exact le_top,
      /-
        🎉 no goals
      -/
                                   /-
                                     ι : Type u_1
                                     α : Type u_2
                                     β : Type u_3
                                     inst✝⁶ : Max α
                                     inst✝⁵ : Min α
                                     inst✝⁴ : Top α
                                     inst✝³ : Bot α
                                     inst✝² : HNot α
                                     inst✝¹ : SDiff α
                                     inst✝ : CoheytingAlgebra β
                                     f : α → β
                                     hf : Function.Injective f
                                     map_sup : ∀ (a b : α), Eq (f (Max.max a b)) (Max.max (f a) (f b))
                                     map_inf : ∀ (a b : α), Eq (f (Min.min a b)) (Min.min (f a) (f b))
                                     map_top : Eq (f Top.top) Top.top
                                     map_bot : Eq (f Bot.bot) Bot.bot
                                     map_hnot : ∀ (a : α), Eq (f (HNot.hnot a)) (HNot.hnot (f a))
                                     map_sdiff : ∀ (a b : α), Eq (f (SDiff.sdiff a b)) (SDiff.sdiff (f a) (f b))
                                     a : α
                                     ⊢ Eq (f (SDiff.sdiff Top.top a)) (f (HNot.hnot a))
                                   -/
    top_sdiff := fun a => hf <| by rw [map_sdiff, map_hnot, map_top, top_sdiff'] }
                                   /-
                                     🎉 no goals
                                   -/

-- See note [reducible non-instances]

/-- Pullback a `BiheytingAlgebra` along an injection. -/
protected abbrev Function.Injective.biheytingAlgebra [Max α] [Min α] [Top α] [Bot α]
    [HasCompl α] [HNot α] [HImp α] [SDiff α] [BiheytingAlgebra β] (f : α → β)
    (hf : Injective f) (map_sup : ∀ a b, f (a ⊔ b) = f a ⊔ f b)
    (map_inf : ∀ a b, f (a ⊓ b) = f a ⊓ f b) (map_top : f ⊤ = ⊤) (map_bot : f ⊥ = ⊥)
    (map_compl : ∀ a, f aᶜ = (f a)ᶜ) (map_hnot : ∀ a, f (￢a) = ￢f a)
    (map_himp : ∀ a b, f (a ⇨ b) = f a ⇨ f b) (map_sdiff : ∀ a b, f (a \ b) = f a \ f b) :
    BiheytingAlgebra α :=
  { hf.heytingAlgebra f map_sup map_inf map_top map_bot map_compl map_himp,
    hf.coheytingAlgebra f map_sup map_inf map_top map_bot map_hnot map_sdiff with }


instance instBiheytingAlgebra : BiheytingAlgebra PUnit.{u+1} :=
  { PUnit.instLinearOrder.{u} with
    top := unit,
    bot := unit,
    sup := fun _ _ => unit,
    inf := fun _ _ => unit,
    compl := fun _ => unit,
    sdiff := fun _ _ => unit,
    hnot := fun _ => unit,
    himp := fun _ _ => unit,
    le_top := fun _ => trivial,
    le_sup_left := fun _ _ => trivial,
    le_sup_right := fun _ _ => trivial,
    sup_le := fun _ _ _ _ _ => trivial,
    inf_le_left := fun _ _ => trivial,
    inf_le_right := fun _ _ => trivial,
    le_inf := fun _ _ _ _ _ => trivial,
    bot_le := fun _ => trivial,
    le_himp_iff := fun _ _ _ => Iff.rfl,
    himp_bot := fun _ => rfl,
    top_sdiff := fun _ => rfl,
    sdiff_le_iff := fun _ _ _ => Iff.rfl }


@[simp]
theorem top_eq : (⊤ : PUnit) = unit :=
  rfl


@[simp]
theorem bot_eq : (⊥ : PUnit) = unit :=
  rfl


@[simp]
theorem sup_eq : a ⊔ b = unit :=
  rfl


@[simp]
theorem inf_eq : a ⊓ b = unit :=
  rfl


@[simp]
theorem compl_eq : aᶜ = unit :=
  rfl


@[simp]
theorem sdiff_eq : a \ b = unit :=
  rfl


@[simp]
theorem hnot_eq : ￢a = unit :=
  rfl


@[simp]
theorem himp_eq : a ⇨ b = unit :=
  rfl


