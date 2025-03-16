/-- A weakly upper modular lattice is a lattice where `a ⊔ b` covers `a` and `b` if `a` and `b` both
cover `a ⊓ b`. -/
class IsWeakUpperModularLattice (α : Type*) [Lattice α] : Prop where
/-- `a ⊔ b` covers `a` and `b` if `a` and `b` both cover `a ⊓ b`. -/
  covBy_sup_of_inf_covBy_covBy {a b : α} : a ⊓ b ⋖ a → a ⊓ b ⋖ b → a ⋖ a ⊔ b


/-- A weakly lower modular lattice is a lattice where `a` and `b` cover `a ⊓ b` if `a ⊔ b` covers
both `a` and `b`. -/
class IsWeakLowerModularLattice (α : Type*) [Lattice α] : Prop where
/-- `a` and `b` cover `a ⊓ b` if `a ⊔ b` covers both `a` and `b` -/
  inf_covBy_of_covBy_covBy_sup {a b : α} : a ⋖ a ⊔ b → b ⋖ a ⊔ b → a ⊓ b ⋖ a


/-- An upper modular lattice, aka semimodular lattice, is a lattice where `a ⊔ b` covers `a` and `b`
if either `a` or `b` covers `a ⊓ b`. -/
class IsUpperModularLattice (α : Type*) [Lattice α] : Prop where
/-- `a ⊔ b` covers `a` and `b` if either `a` or `b` covers `a ⊓ b` -/
  covBy_sup_of_inf_covBy {a b : α} : a ⊓ b ⋖ a → b ⋖ a ⊔ b


/-- A lower modular lattice is a lattice where `a` and `b` both cover `a ⊓ b` if `a ⊔ b` covers
either `a` or `b`. -/
class IsLowerModularLattice (α : Type*) [Lattice α] : Prop where
/-- `a` and `b` both cover `a ⊓ b` if `a ⊔ b` covers either `a` or `b` -/
  inf_covBy_of_covBy_sup {a b : α} : a ⋖ a ⊔ b → a ⊓ b ⋖ b


/-- A modular lattice is one with a limited associativity between `⊓` and `⊔`. -/
class IsModularLattice (α : Type*) [Lattice α] : Prop where
/-- Whenever `x ≤ z`, then for any `y`, `(x ⊔ y) ⊓ z ≤ x ⊔ (y ⊓ z)`  -/
  sup_inf_le_assoc_of_le : ∀ {x : α} (y : α) {z : α}, x ≤ z → (x ⊔ y) ⊓ z ≤ x ⊔ y ⊓ z


theorem covBy_sup_of_inf_covBy_of_inf_covBy_left : a ⊓ b ⋖ a → a ⊓ b ⋖ b → a ⋖ a ⊔ b :=
  IsWeakUpperModularLattice.covBy_sup_of_inf_covBy_covBy


theorem covBy_sup_of_inf_covBy_of_inf_covBy_right : a ⊓ b ⋖ a → a ⊓ b ⋖ b → b ⋖ a ⊔ b := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsWeakUpperModularLattice α
    a b : α
    ⊢ CovBy (Min.min a b) a → CovBy (Min.min a b) b → CovBy b (Max.max a b)
  -/
  rw [inf_comm, sup_comm]
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsWeakUpperModularLattice α
    a b : α
    ⊢ CovBy (Min.min b a) a → CovBy (Min.min b a) b → CovBy b (Max.max b a)
  -/
  exact fun ha hb => covBy_sup_of_inf_covBy_of_inf_covBy_left hb ha
  /-
    🎉 no goals
  -/


alias CovBy.sup_of_inf_of_inf_left := covBy_sup_of_inf_covBy_of_inf_covBy_left


alias CovBy.sup_of_inf_of_inf_right := covBy_sup_of_inf_covBy_of_inf_covBy_right


instance : IsWeakLowerModularLattice (OrderDual α) :=
  ⟨fun ha hb => (ha.ofDual.sup_of_inf_of_inf_left hb.ofDual).toDual⟩


theorem inf_covBy_of_covBy_sup_of_covBy_sup_left : a ⋖ a ⊔ b → b ⋖ a ⊔ b → a ⊓ b ⋖ a :=
  IsWeakLowerModularLattice.inf_covBy_of_covBy_covBy_sup


theorem inf_covBy_of_covBy_sup_of_covBy_sup_right : a ⋖ a ⊔ b → b ⋖ a ⊔ b → a ⊓ b ⋖ b := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsWeakLowerModularLattice α
    a b : α
    ⊢ CovBy a (Max.max a b) → CovBy b (Max.max a b) → CovBy (Min.min a b) b
  -/
  rw [sup_comm, inf_comm]
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsWeakLowerModularLattice α
    a b : α
    ⊢ CovBy a (Max.max b a) → CovBy b (Max.max b a) → CovBy (Min.min b a) b
  -/
  exact fun ha hb => inf_covBy_of_covBy_sup_of_covBy_sup_left hb ha
  /-
    🎉 no goals
  -/


alias CovBy.inf_of_sup_of_sup_left := inf_covBy_of_covBy_sup_of_covBy_sup_left


alias CovBy.inf_of_sup_of_sup_right := inf_covBy_of_covBy_sup_of_covBy_sup_right


instance : IsWeakUpperModularLattice (OrderDual α) :=
  ⟨fun ha hb => (ha.ofDual.inf_of_sup_of_sup_left hb.ofDual).toDual⟩


theorem covBy_sup_of_inf_covBy_left : a ⊓ b ⋖ a → b ⋖ a ⊔ b :=
  IsUpperModularLattice.covBy_sup_of_inf_covBy


theorem covBy_sup_of_inf_covBy_right : a ⊓ b ⋖ b → a ⋖ a ⊔ b := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsUpperModularLattice α
    a b : α
    ⊢ CovBy (Min.min a b) b → CovBy a (Max.max a b)
  -/
  rw [sup_comm, inf_comm]
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsUpperModularLattice α
    a b : α
    ⊢ CovBy (Min.min b a) b → CovBy a (Max.max b a)
  -/
  exact covBy_sup_of_inf_covBy_left
  /-
    🎉 no goals
  -/


alias CovBy.sup_of_inf_left := covBy_sup_of_inf_covBy_left


alias CovBy.sup_of_inf_right := covBy_sup_of_inf_covBy_right

-- See note [lower instance priority]

instance (priority := 100) IsUpperModularLattice.to_isWeakUpperModularLattice :
    IsWeakUpperModularLattice α :=
  ⟨fun _ => CovBy.sup_of_inf_right⟩


instance : IsLowerModularLattice (OrderDual α) :=
  ⟨fun h => h.ofDual.sup_of_inf_left.toDual⟩


theorem inf_covBy_of_covBy_sup_left : a ⋖ a ⊔ b → a ⊓ b ⋖ b :=
  IsLowerModularLattice.inf_covBy_of_covBy_sup


theorem inf_covBy_of_covBy_sup_right : b ⋖ a ⊔ b → a ⊓ b ⋖ a := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsLowerModularLattice α
    a b : α
    ⊢ CovBy b (Max.max a b) → CovBy (Min.min a b) a
  -/
  rw [inf_comm, sup_comm]
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsLowerModularLattice α
    a b : α
    ⊢ CovBy b (Max.max b a) → CovBy (Min.min b a) a
  -/
  exact inf_covBy_of_covBy_sup_left
  /-
    🎉 no goals
  -/


alias CovBy.inf_of_sup_left := inf_covBy_of_covBy_sup_left


alias CovBy.inf_of_sup_right := inf_covBy_of_covBy_sup_right

-- See note [lower instance priority]

instance (priority := 100) IsLowerModularLattice.to_isWeakLowerModularLattice :
    IsWeakLowerModularLattice α :=
  ⟨fun _ => CovBy.inf_of_sup_right⟩


instance : IsUpperModularLattice (OrderDual α) :=
  ⟨fun h => h.ofDual.inf_of_sup_left.toDual⟩


theorem sup_inf_assoc_of_le {x : α} (y : α) {z : α} (h : x ≤ z) : (x ⊔ y) ⊓ z = x ⊔ y ⊓ z :=
  le_antisymm (IsModularLattice.sup_inf_le_assoc_of_le y h)
    (le_inf (sup_le_sup_left inf_le_left _) (sup_le h inf_le_right))


theorem IsModularLattice.inf_sup_inf_assoc {x y z : α} : x ⊓ z ⊔ y ⊓ z = (x ⊓ z ⊔ y) ⊓ z :=
  (sup_inf_assoc_of_le y inf_le_right).symm


theorem inf_sup_assoc_of_le {x : α} (y : α) {z : α} (h : z ≤ x) : x ⊓ y ⊔ z = x ⊓ (y ⊔ z) := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsModularLattice α
    x y z : α
    h : LE.le z x
    ⊢ Eq (Max.max (Min.min x y) z) (Min.min x (Max.max y z))
  -/
  rw [inf_comm, sup_comm, ← sup_inf_assoc_of_le y h, inf_comm, sup_comm]
  /-
    🎉 no goals
  -/


instance : IsModularLattice αᵒᵈ :=
  ⟨fun y z xz =>
    le_of_eq
      (by
        /-
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : IsModularLattice α
          x✝ y z : OrderDual α
          xz : LE.le x✝ z
          ⊢ Eq (Min.min (Max.max x✝ y) z) (Max.max x✝ (Min.min y z))
        -/
        rw [inf_comm, sup_comm, eq_comm, inf_comm, sup_comm]
        /-
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : IsModularLattice α
          x✝ y z : OrderDual α
          xz : LE.le x✝ z
          ⊢ Eq (Max.max (Min.min z y) x✝) (Min.min z (Max.max y x✝))
        -/
        exact @sup_inf_assoc_of_le α _ _ _ y _ xz)⟩
        /-
          🎉 no goals
        -/


theorem IsModularLattice.sup_inf_sup_assoc : (x ⊔ z) ⊓ (y ⊔ z) = (x ⊔ z) ⊓ y ⊔ z :=
  @IsModularLattice.inf_sup_inf_assoc αᵒᵈ _ _ _ _ _


theorem eq_of_le_of_inf_le_of_le_sup (hxy : x ≤ y) (hinf : y ⊓ z ≤ x) (hsup : y ≤ x ⊔ z) :
    x = y := by
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsModularLattice α
    x y z : α
    hxy : LE.le x y
    hinf : LE.le (Min.min y z) x
    hsup : LE.le y (Max.max x z)
    ⊢ Eq x y
  -/
  refine hxy.antisymm ?_
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsModularLattice α
    x y z : α
    hxy : LE.le x y
    hinf : LE.le (Min.min y z) x
    hsup : LE.le y (Max.max x z)
    ⊢ LE.le y x
  -/
  rw [← inf_eq_right, sup_inf_assoc_of_le _ hxy] at hsup
  /-
    α : Type u_1
    inst✝¹ : Lattice α
    inst✝ : IsModularLattice α
    x y z : α
    hxy : LE.le x y
    hinf : LE.le (Min.min y z) x
    hsup : Eq (Max.max x (Min.min z y)) y
    ⊢ LE.le y x
  -/
  rwa [← hsup, sup_le_iff, and_iff_right rfl.le, inf_comm]
  /-
    🎉 no goals
  -/


theorem eq_of_le_of_inf_le_of_sup_le (hxy : x ≤ y) (hinf : y ⊓ z ≤ x ⊓ z) (hsup : y ⊔ z ≤ x ⊔ z) :
    x = y :=
  eq_of_le_of_inf_le_of_le_sup hxy (hinf.trans inf_le_left) (le_sup_left.trans hsup)


theorem sup_lt_sup_of_lt_of_inf_le_inf (hxy : x < y) (hinf : y ⊓ z ≤ x ⊓ z) : x ⊔ z < y ⊔ z :=
  lt_of_le_of_ne (sup_le_sup_right (le_of_lt hxy) _) fun hsup =>
    ne_of_lt hxy <| eq_of_le_of_inf_le_of_sup_le (le_of_lt hxy) hinf (le_of_eq hsup.symm)


theorem inf_lt_inf_of_lt_of_sup_le_sup (hxy : x < y) (hinf : y ⊔ z ≤ x ⊔ z) : x ⊓ z < y ⊓ z :=
  sup_lt_sup_of_lt_of_inf_le_inf (α := αᵒᵈ) hxy hinf


theorem strictMono_inf_prod_sup : StrictMono fun x ↦ (x ⊓ z, x ⊔ z) := fun _x _y hxy ↦
  ⟨⟨inf_le_inf_right _ hxy.le, sup_le_sup_right hxy.le _⟩,
    fun ⟨inf_le, sup_le⟩ ↦ (sup_lt_sup_of_lt_of_inf_le_inf hxy inf_le).not_le sup_le⟩


/-- A generalization of the theorem that if `N` is a submodule of `M` and
  `N` and `M / N` are both Artinian, then `M` is Artinian. -/
theorem wellFounded_lt_exact_sequence {β γ : Type*} [PartialOrder β] [Preorder γ]
    [h₁ : WellFoundedLT β] [h₂ : WellFoundedLT γ] (K : α)
    (f₁ : β → α) (f₂ : α → β) (g₁ : γ → α) (g₂ : α → γ) (gci : GaloisCoinsertion f₁ f₂)
    (gi : GaloisInsertion g₂ g₁) (hf : ∀ a, f₁ (f₂ a) = a ⊓ K) (hg : ∀ a, g₁ (g₂ a) = a ⊔ K) :
    WellFoundedLT α :=
  StrictMono.wellFoundedLT (f := fun A ↦ (f₂ A, g₂ A)) fun A B hAB ↦ by
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : IsModularLattice α
      β : Type u_2
      γ : Type u_3
      inst✝¹ : PartialOrder β
      inst✝ : Preorder γ
      h₁ : WellFoundedLT β
      h₂ : WellFoundedLT γ
      K : α
      f₁ : β → α
      f₂ : α → β
      g₁ : γ → α
      g₂ : α → γ
      gci : GaloisCoinsertion f₁ f₂
      gi : GaloisInsertion g₂ g₁
      hf : ∀ (a : α), Eq (f₁ (f₂ a)) (Min.min a K)
      hg : ∀ (a : α), Eq (g₁ (g₂ a)) (Max.max a K)
      A B : α
      hAB : LT.lt A B
      ⊢ LT.lt ((fun A => { fst := f₂ A, snd := g₂ A }) A) ((fun A => { fst := f₂ A,  …
    -/
    simp only [Prod.le_def, lt_iff_le_not_le, ← gci.l_le_l_iff, ← gi.u_le_u_iff, hf, hg]
    /-
      α : Type u_1
      inst✝³ : Lattice α
      inst✝² : IsModularLattice α
      β : Type u_2
      γ : Type u_3
      inst✝¹ : PartialOrder β
      inst✝ : Preorder γ
      h₁ : WellFoundedLT β
      h₂ : WellFoundedLT γ
      K : α
      f₁ : β → α
      f₂ : α → β
      g₁ : γ → α
      g₂ : α → γ
      gci : GaloisCoinsertion f₁ f₂
      gi : GaloisInsertion g₂ g₁
      hf : ∀ (a : α), Eq (f₁ (f₂ a)) (Min.min a K)
      hg : ∀ (a : α), Eq (g₁ (g₂ a)) (Max.max a K)
      A B : α
      hAB : LT.lt A B
      ⊢ And (And (LE.le (Min.min A K) (Min.min B K)) (LE.le (Max.max A K) (Max.max B …
    -/
    exact strictMono_inf_prod_sup hAB
    /-
      🎉 no goals
    -/


/-- A generalization of the theorem that if `N` is a submodule of `M` and
  `N` and `M / N` are both Noetherian, then `M` is Noetherian. -/
theorem wellFounded_gt_exact_sequence {β γ : Type*} [Preorder β] [PartialOrder γ]
    [WellFoundedGT β] [WellFoundedGT γ] (K : α)
    (f₁ : β → α) (f₂ : α → β) (g₁ : γ → α) (g₂ : α → γ) (gci : GaloisCoinsertion f₁ f₂)
    (gi : GaloisInsertion g₂ g₁) (hf : ∀ a, f₁ (f₂ a) = a ⊓ K) (hg : ∀ a, g₁ (g₂ a) = a ⊔ K) :
    WellFoundedGT α :=
  wellFounded_lt_exact_sequence (α := αᵒᵈ) (β := γᵒᵈ) (γ := βᵒᵈ)
    K g₁ g₂ f₁ f₂ gi.dual gci.dual hg hf


/-- The diamond isomorphism between the intervals `[a ⊓ b, a]` and `[b, a ⊔ b]` -/
@[simps]
def infIccOrderIsoIccSup (a b : α) : Set.Icc (a ⊓ b) a ≃o Set.Icc b (a ⊔ b) where
  toFun x := ⟨x ⊔ b, ⟨le_sup_right, sup_le_sup_right x.prop.2 b⟩⟩
  invFun x := ⟨a ⊓ x, ⟨inf_le_inf_left a x.prop.1, inf_le_left⟩⟩
  left_inv x :=
    Subtype.ext
      (by
        /-
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : IsModularLattice α
          x✝ y z a b : α
          x : ↑(Set.Icc (Min.min a b) a)
          ⊢ Eq ↑((fun x => ⟨Min.min a ↑x, ⋯⟩) ((fun x => ⟨Max.max (↑x) b, ⋯⟩) x)) ↑x
        -/
        change a ⊓ (↑x ⊔ b) = ↑x
        /-
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : IsModularLattice α
          x✝ y z a b : α
          x : ↑(Set.Icc (Min.min a b) a)
          ⊢ Eq (Min.min a (Max.max (↑x) b)) ↑x
        -/
        rw [sup_comm, ← inf_sup_assoc_of_le _ x.prop.2, sup_eq_right.2 x.prop.1])
        /-
          🎉 no goals
        -/
  right_inv x :=
    Subtype.ext
      (by
        /-
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : IsModularLattice α
          x✝ y z a b : α
          x : ↑(Set.Icc b (Max.max a b))
          ⊢ Eq ↑((fun x => ⟨Max.max (↑x) b, ⋯⟩) ((fun x => ⟨Min.min a ↑x, ⋯⟩) x)) ↑x
        -/
        change a ⊓ ↑x ⊔ b = ↑x
        /-
          α : Type u_1
          inst✝¹ : Lattice α
          inst✝ : IsModularLattice α
          x✝ y z a b : α
          x : ↑(Set.Icc b (Max.max a b))
          ⊢ Eq (Max.max (Min.min a ↑x) b) ↑x
        -/
        rw [inf_comm, inf_sup_assoc_of_le _ x.prop.1, inf_eq_left.2 x.prop.2])
        /-
          🎉 no goals
        -/
  map_rel_iff' {x y} := by
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : IsModularLattice α
      x✝ y✝ z a b : α
      x y : ↑(Set.Icc (Min.min a b) a)
      ⊢ Iff (LE.le ({ toFun := fun x => ⟨Max.max (↑x) b, ⋯⟩, invFun := fun x => ⟨Min …
    -/
    simp only [Subtype.mk_le_mk, Equiv.coe_fn_mk, le_sup_right]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : IsModularLattice α
      x✝ y✝ z a b : α
      x y : ↑(Set.Icc (Min.min a b) a)
      ⊢ Iff (LE.le (Max.max (↑x) b) (Max.max (↑y) b)) (LE.le x y)
    -/
    rw [← Subtype.coe_le_coe]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : IsModularLattice α
      x✝ y✝ z a b : α
      x y : ↑(Set.Icc (Min.min a b) a)
      ⊢ Iff (LE.le (Max.max (↑x) b) (Max.max (↑y) b)) (LE.le ↑x ↑y)
    -/
    refine ⟨fun h => ?_, fun h => sup_le_sup_right h _⟩
    rw [← sup_eq_right.2 x.prop.1, inf_sup_assoc_of_le _ x.prop.2, sup_comm, ←
      sup_eq_right.2 y.prop.1, inf_sup_assoc_of_le _ y.prop.2, sup_comm b]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : IsModularLattice α
      x✝ y✝ z a b : α
      x y : ↑(Set.Icc (Min.min a b) a)
      h : LE.le (Max.max (↑x) b) (Max.max (↑y) b)
      ⊢ LE.le (Min.min a (Max.max (↑x) b)) (Min.min a (Max.max (↑y) b))
    -/
    exact inf_le_inf_left _ h
    /-
      🎉 no goals
    -/


theorem inf_strictMonoOn_Icc_sup {a b : α} : StrictMonoOn (fun c => a ⊓ c) (Icc b (a ⊔ b)) :=
  StrictMono.of_restrict (infIccOrderIsoIccSup a b).symm.strictMono


theorem sup_strictMonoOn_Icc_inf {a b : α} : StrictMonoOn (fun c => c ⊔ b) (Icc (a ⊓ b) a) :=
  StrictMono.of_restrict (infIccOrderIsoIccSup a b).strictMono


/-- The diamond isomorphism between the intervals `]a ⊓ b, a[` and `}b, a ⊔ b[`. -/
@[simps]
def infIooOrderIsoIooSup (a b : α) : Ioo (a ⊓ b) a ≃o Ioo b (a ⊔ b) where
  toFun c :=
    ⟨c ⊔ b,
      le_sup_right.trans_lt <|
        sup_strictMonoOn_Icc_inf (left_mem_Icc.2 inf_le_left) (Ioo_subset_Icc_self c.2) c.2.1,
      sup_strictMonoOn_Icc_inf (Ioo_subset_Icc_self c.2) (right_mem_Icc.2 inf_le_left) c.2.2⟩
  invFun c :=
    ⟨a ⊓ c,
      inf_strictMonoOn_Icc_sup (left_mem_Icc.2 le_sup_right) (Ioo_subset_Icc_self c.2) c.2.1,
      inf_le_left.trans_lt' <|
        inf_strictMonoOn_Icc_sup (Ioo_subset_Icc_self c.2) (right_mem_Icc.2 le_sup_right) c.2.2⟩
  left_inv c :=
    Subtype.ext <| by
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : IsModularLattice α
        x y z a b : α
        c : ↑(Set.Ioo (Min.min a b) a)
        ⊢ Eq ↑((fun c => ⟨Min.min a ↑c, ⋯⟩) ((fun c => ⟨Max.max (↑c) b, ⋯⟩) c)) ↑c
      -/
      dsimp
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : IsModularLattice α
        x y z a b : α
        c : ↑(Set.Ioo (Min.min a b) a)
        ⊢ Eq (Min.min a (Max.max (↑c) b)) ↑c
      -/
      rw [sup_comm, ← inf_sup_assoc_of_le _ c.prop.2.le, sup_eq_right.2 c.prop.1.le]
      /-
        🎉 no goals
      -/
  right_inv c :=
    Subtype.ext <| by
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : IsModularLattice α
        x y z a b : α
        c : ↑(Set.Ioo b (Max.max a b))
        ⊢ Eq ↑((fun c => ⟨Max.max (↑c) b, ⋯⟩) ((fun c => ⟨Min.min a ↑c, ⋯⟩) c)) ↑c
      -/
      dsimp
      /-
        α : Type u_1
        inst✝¹ : Lattice α
        inst✝ : IsModularLattice α
        x y z a b : α
        c : ↑(Set.Ioo b (Max.max a b))
        ⊢ Eq (Max.max (Min.min a ↑c) b) ↑c
      -/
      rw [inf_comm, inf_sup_assoc_of_le _ c.prop.1.le, inf_eq_left.2 c.prop.2.le]
      /-
        🎉 no goals
      -/
  map_rel_iff' := @fun c d =>
    @OrderIso.le_iff_le _ _ _ _ (infIccOrderIsoIccSup _ _) ⟨c.1, Ioo_subset_Icc_self c.2⟩
      ⟨d.1, Ioo_subset_Icc_self d.2⟩

-- See note [lower instance priority]

instance (priority := 100) IsModularLattice.to_isLowerModularLattice : IsLowerModularLattice α :=
  ⟨fun {a b} => by
    simp_rw [covBy_iff_Ioo_eq, sup_comm a, inf_comm a, ← isEmpty_coe_sort, right_lt_sup,
      inf_lt_left, (infIooOrderIsoIooSup b a).symm.toEquiv.isEmpty_congr]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : IsModularLattice α
      x y z a b : α
      ⊢ And (Not (LE.le b a)) (IsEmpty ↑(Set.Ioo (Min.min b a) b)) → And (Not (LE.le …
    -/
    exact id⟩
    /-
      🎉 no goals
    -/

-- See note [lower instance priority]

instance (priority := 100) IsModularLattice.to_isUpperModularLattice : IsUpperModularLattice α :=
  ⟨fun {a b} => by
    simp_rw [covBy_iff_Ioo_eq, ← isEmpty_coe_sort, right_lt_sup, inf_lt_left,
      (infIooOrderIsoIooSup a b).toEquiv.isEmpty_congr]
    /-
      α : Type u_1
      inst✝¹ : Lattice α
      inst✝ : IsModularLattice α
      x y z a b : α
      ⊢ And (Not (LE.le a b)) (IsEmpty ↑(Set.Ioo b (Max.max a b))) → And (Not (LE.le …
    -/
    exact id⟩
    /-
      🎉 no goals
    -/


/-- The diamond isomorphism between the intervals `Set.Iic a` and `Set.Ici b`. -/
def IicOrderIsoIci {a b : α} (h : IsCompl a b) : Set.Iic a ≃o Set.Ici b :=
  (OrderIso.setCongr (Set.Iic a) (Set.Icc (a ⊓ b) a)
        (h.inf_eq_bot.symm ▸ Set.Icc_bot.symm)).trans <|
    (infIccOrderIsoIccSup a b).trans
      (OrderIso.setCongr (Set.Icc b (a ⊔ b)) (Set.Ici b) (h.sup_eq_top.symm ▸ Set.Icc_top))


theorem isModularLattice_iff_inf_sup_inf_assoc [Lattice α] :
    IsModularLattice α ↔ ∀ x y z : α, x ⊓ z ⊔ y ⊓ z = (x ⊓ z ⊔ y) ⊓ z :=
  ⟨fun h => @IsModularLattice.inf_sup_inf_assoc _ _ h, fun h =>
                      /-
                        α : Type u_1
                        inst✝ : Lattice α
                        h : ∀ (x y z : α), Eq (Max.max (Min.min x z) (Min.min y z)) (Min.min (Max.max  …
                        x✝ y z : α
                        xz : LE.le x✝ z
                        ⊢ LE.le (Min.min (Max.max x✝ y) z) (Max.max x✝ (Min.min y z))
                      -/
    ⟨fun y z xz => by rw [← inf_eq_left.2 xz, h]⟩⟩
                      /-
                        🎉 no goals
                      -/


instance (priority := 100) [DistribLattice α] : IsModularLattice α :=
                    /-
                      α : Type u_1
                      inst✝ : DistribLattice α
                      x✝ y z : α
                      xz : LE.le x✝ z
                      ⊢ LE.le (Min.min (Max.max x✝ y) z) (Max.max x✝ (Min.min y z))
                    -/
  ⟨fun y z xz => by rw [inf_sup_right, inf_eq_left.2 xz]⟩
                    /-
                      🎉 no goals
                    -/


theorem disjoint_sup_right_of_disjoint_sup_left [Lattice α] [OrderBot α]
    [IsModularLattice α] (h : Disjoint a b) (hsup : Disjoint (a ⊔ b) c) :
    Disjoint a (b ⊔ c) := by
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint a b
    hsup : Disjoint (Max.max a b) c
    ⊢ Disjoint a (Max.max b c)
  -/
  rw [disjoint_iff_inf_le, ← h.eq_bot, sup_comm]
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint a b
    hsup : Disjoint (Max.max a b) c
    ⊢ LE.le (Min.min a (Max.max c b)) (Min.min a b)
  -/
  apply le_inf inf_le_left
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint a b
    hsup : Disjoint (Max.max a b) c
    ⊢ LE.le (Min.min a (Max.max c b)) b
  -/
  apply (inf_le_inf_right (c ⊔ b) le_sup_right).trans
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint a b
    hsup : Disjoint (Max.max a b) c
    ⊢ LE.le (Min.min (Max.max ?m.25398 a) (Max.max c b)) b
  -/
  rw [sup_comm, IsModularLattice.sup_inf_sup_assoc, hsup.eq_bot, bot_sup_eq]
  /-
    🎉 no goals
  -/


theorem disjoint_sup_left_of_disjoint_sup_right [Lattice α] [OrderBot α]
    [IsModularLattice α] (h : Disjoint b c) (hsup : Disjoint a (b ⊔ c)) :
    Disjoint (a ⊔ b) c := by
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint b c
    hsup : Disjoint a (Max.max b c)
    ⊢ Disjoint (Max.max a b) c
  -/
  rw [disjoint_comm, sup_comm]
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint b c
    hsup : Disjoint a (Max.max b c)
    ⊢ Disjoint c (Max.max b a)
  -/
  apply Disjoint.disjoint_sup_right_of_disjoint_sup_left h.symm
  /-
    α : Type u_1
    a b c : α
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    inst✝ : IsModularLattice α
    h : Disjoint b c
    hsup : Disjoint a (Max.max b c)
    ⊢ Disjoint (Max.max c b) a
  -/
  rwa [sup_comm, disjoint_comm] at hsup
  /-
    🎉 no goals
  -/


theorem isCompl_sup_right_of_isCompl_sup_left [Lattice α] [BoundedOrder α] [IsModularLattice α]
    (h : Disjoint a b) (hcomp : IsCompl (a ⊔ b) c) :
    IsCompl a (b ⊔ c) :=
  ⟨h.disjoint_sup_right_of_disjoint_sup_left hcomp.disjoint, codisjoint_assoc.mp hcomp.codisjoint⟩


theorem isCompl_sup_left_of_isCompl_sup_right [Lattice α] [BoundedOrder α] [IsModularLattice α]
    (h : Disjoint b c) (hcomp : IsCompl a (b ⊔ c)) :
    IsCompl (a ⊔ b) c :=
  ⟨h.disjoint_sup_left_of_disjoint_sup_right hcomp.disjoint, codisjoint_assoc.mpr hcomp.codisjoint⟩


lemma Set.Iic.isCompl_inf_inf_of_isCompl_of_le [Lattice α] [BoundedOrder α] [IsModularLattice α]
    {a b c : α} (h₁ : IsCompl b c) (h₂ : b ≤ a) :
    IsCompl (⟨a ⊓ b, inf_le_left⟩ : Iic a) (⟨a ⊓ c, inf_le_left⟩ : Iic a) := by
  /-
    α : Type u_1
    inst✝² : Lattice α
    inst✝¹ : BoundedOrder α
    inst✝ : IsModularLattice α
    a b c : α
    h₁ : IsCompl b c
    h₂ : LE.le b a
    ⊢ IsCompl ⟨Min.min a b, ⋯⟩ ⟨Min.min a c, ⋯⟩
  -/
  constructor
    /-
      case disjoint
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : BoundedOrder α
      inst✝ : IsModularLattice α
      a b c : α
      h₁ : IsCompl b c
      h₂ : LE.le b a
      ⊢ Disjoint ⟨Min.min a b, ⋯⟩ ⟨Min.min a c, ⋯⟩
    -/
  · simp [disjoint_iff, Subtype.ext_iff, inf_comm a c, inf_assoc a, ← inf_assoc b, h₁.inf_eq_bot]
    /-
      🎉 no goals
    -/
    /-
      case codisjoint
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : BoundedOrder α
      inst✝ : IsModularLattice α
      a b c : α
      h₁ : IsCompl b c
      h₂ : LE.le b a
      ⊢ Codisjoint ⟨Min.min a b, ⋯⟩ ⟨Min.min a c, ⋯⟩
    -/
  · simp only [Iic.codisjoint_iff, inf_comm a, IsModularLattice.inf_sup_inf_assoc]
    /-
      case codisjoint
      α : Type u_1
      inst✝² : Lattice α
      inst✝¹ : BoundedOrder α
      inst✝ : IsModularLattice α
      a b c : α
      h₁ : IsCompl b c
      h₂ : LE.le b a
      ⊢ Eq (Min.min (Max.max (Min.min b a) c) a) a
    -/
    simp [inf_of_le_left h₂, h₁.sup_eq_top]
    /-
      🎉 no goals
    -/


instance isModularLattice_Iic : IsModularLattice (Set.Iic a) :=
  ⟨@fun x y z xz => (sup_inf_le_assoc_of_le (y : α) xz : (↑x ⊔ ↑y) ⊓ ↑z ≤ ↑x ⊔ ↑y ⊓ ↑z)⟩


instance isModularLattice_Ici : IsModularLattice (Set.Ici a) :=
  ⟨@fun x y z xz => (sup_inf_le_assoc_of_le (y : α) xz : (↑x ⊔ ↑y) ⊓ ↑z ≤ ↑x ⊔ ↑y ⊓ ↑z)⟩


instance complementedLattice_Iic : ComplementedLattice (Set.Iic a) :=
  ⟨fun ⟨x, hx⟩ =>
    let ⟨y, hy⟩ := exists_isCompl x
    ⟨⟨y ⊓ a, Set.mem_Iic.2 inf_le_right⟩, by
      /-
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : IsModularLattice α
        a : α
        inst✝¹ : BoundedOrder α
        inst✝ : ComplementedLattice α
        x✝ : ↑(Set.Iic a)
        x : α
        hx : Membership.mem (Set.Iic a) x
        y : α
        hy : IsCompl x y
        ⊢ IsCompl ⟨x, hx⟩ ⟨Min.min y a, ⋯⟩
      -/
      constructor
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ Disjoint ⟨x, hx⟩ ⟨Min.min y a, ⋯⟩
        -/
      · rw [disjoint_iff_inf_le]
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le (Min.min ⟨x, hx⟩ ⟨Min.min y a, ⋯⟩) Bot.bot
        -/
        change x ⊓ (y ⊓ a) ≤ ⊥
        -- improve lattice subtype API
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le (Min.min x (Min.min y a)) Bot.bot
        -/
        rw [← inf_assoc]
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le (Min.min (Min.min x y) a) Bot.bot
        -/
        exact le_trans inf_le_left hy.1.le_bot
        /-
          🎉 no goals
        -/
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ Codisjoint ⟨x, hx⟩ ⟨Min.min y a, ⋯⟩
        -/
      · rw [codisjoint_iff_le_sup]
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le Top.top (Max.max ⟨x, hx⟩ ⟨Min.min y a, ⋯⟩)
        -/
        change a ≤ x ⊔ y ⊓ a
        -- improve lattice subtype API
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Iic a)
          x : α
          hx : Membership.mem (Set.Iic a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le a (Max.max x (Min.min y a))
        -/
        rw [← sup_inf_assoc_of_le _ (Set.mem_Iic.1 hx), hy.2.eq_top, top_inf_eq]⟩⟩
        /-
          🎉 no goals
        -/


instance complementedLattice_Ici : ComplementedLattice (Set.Ici a) :=
  ⟨fun ⟨x, hx⟩ =>
    let ⟨y, hy⟩ := exists_isCompl x
    ⟨⟨y ⊔ a, Set.mem_Ici.2 le_sup_right⟩, by
      /-
        α : Type u_1
        inst✝³ : Lattice α
        inst✝² : IsModularLattice α
        a : α
        inst✝¹ : BoundedOrder α
        inst✝ : ComplementedLattice α
        x✝ : ↑(Set.Ici a)
        x : α
        hx : Membership.mem (Set.Ici a) x
        y : α
        hy : IsCompl x y
        ⊢ IsCompl ⟨x, hx⟩ ⟨Max.max y a, ⋯⟩
      -/
      constructor
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ Disjoint ⟨x, hx⟩ ⟨Max.max y a, ⋯⟩
        -/
      · rw [disjoint_iff_inf_le]
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le (Min.min ⟨x, hx⟩ ⟨Max.max y a, ⋯⟩) Bot.bot
        -/
        change x ⊓ (y ⊔ a) ≤ a
        -- improve lattice subtype API
        /-
          case disjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le (Min.min x (Max.max y a)) a
        -/
        rw [← inf_sup_assoc_of_le _ (Set.mem_Ici.1 hx), hy.1.eq_bot, bot_sup_eq]
        /-
          🎉 no goals
        -/
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ Codisjoint ⟨x, hx⟩ ⟨Max.max y a, ⋯⟩
        -/
      · rw [codisjoint_iff_le_sup]
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le Top.top (Max.max ⟨x, hx⟩ ⟨Max.max y a, ⋯⟩)
        -/
        change ⊤ ≤ x ⊔ (y ⊔ a)
        -- improve lattice subtype API
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le Top.top (Max.max x (Max.max y a))
        -/
        rw [← sup_assoc]
        /-
          case codisjoint
          α : Type u_1
          inst✝³ : Lattice α
          inst✝² : IsModularLattice α
          a : α
          inst✝¹ : BoundedOrder α
          inst✝ : ComplementedLattice α
          x✝ : ↑(Set.Ici a)
          x : α
          hx : Membership.mem (Set.Ici a) x
          y : α
          hy : IsCompl x y
          ⊢ LE.le Top.top (Max.max (Max.max x y) a)
        -/
        exact le_trans hy.2.top_le le_sup_left⟩⟩
        /-
          🎉 no goals
        -/


