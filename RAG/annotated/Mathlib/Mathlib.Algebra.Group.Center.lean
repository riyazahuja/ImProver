/-- Conditions for an element to be additively central -/
structure IsAddCentral [Add M] (z : M) : Prop where
  /-- addition commutes -/
  comm (a : M) : z + a = a + z
  /-- associative property for left addition -/
  left_assoc (b c : M) : z + (b + c) = (z + b) + c
  /-- middle associative addition property -/
  mid_assoc (a c : M) : (a + z) + c = a + (z + c)
  /-- associative property for right addition -/
  right_assoc (a b : M) : (a + b) + z = a + (b + z)


/-- Conditions for an element to be multiplicatively central -/
@[to_additive]
structure IsMulCentral [Mul M] (z : M) : Prop where
  /-- multiplication commutes -/
  comm (a : M) : z * a = a * z
  /-- associative property for left multiplication -/
  left_assoc (b c : M) : z * (b * c) = (z * b) * c
  /-- middle associative multiplication property -/
  mid_assoc (a c : M) : (a * z) * c = a * (z * c)
  /-- associative property for right multiplication -/
  right_assoc (a b : M) : (a * b) * z = a * (b * z)


attribute [mk_iff] IsMulCentral IsAddCentral

@[to_additive]
protected theorem left_comm (h : IsMulCentral a) (b c) : a * (b * c) = b * (a * c) := by
  /-
    M : Type u_1
    a : M
    inst✝ : Mul M
    h : IsMulCentral a
    b c : M
    ⊢ Eq (HMul.hMul a (HMul.hMul b c)) (HMul.hMul b (HMul.hMul a c))
  -/
  simp only [h.comm, h.right_assoc]
  /-
    🎉 no goals
  -/

-- cf. `Commute.right_comm`

@[to_additive]
protected theorem right_comm (h : IsMulCentral c) (a b) : a * b * c = a * c * b := by
  /-
    M : Type u_1
    c : M
    inst✝ : Mul M
    h : IsMulCentral c
    a b : M
    ⊢ Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul (HMul.hMul a c) b)
  -/
  simp only [h.right_assoc, h.mid_assoc, h.comm]
  /-
    🎉 no goals
  -/


variable (M) in
/-- The center of a magma. -/
@[to_additive addCenter " The center of an additive magma. "]
def center : Set M :=
  { z | IsMulCentral z }


variable (S) in
/-- The centralizer of a subset of a magma. -/
@[to_additive addCentralizer " The centralizer of a subset of an additive magma. "]
def centralizer : Set M := {c | ∀ m ∈ S, m * c = c * m}

-- Porting note: The `to_additive` version used to be `mem_addCenter` without the iff

@[to_additive mem_addCenter_iff]
theorem mem_center_iff {z : M} : z ∈ center M ↔ IsMulCentral z :=
  Iff.rfl


@[to_additive mem_addCentralizer]
lemma mem_centralizer_iff {c : M} : c ∈ centralizer S ↔ ∀ m ∈ S, m * c = c * m := Iff.rfl


@[to_additive (attr := simp) add_mem_addCenter]
theorem mul_mem_center {z₁ z₂ : M} (hz₁ : z₁ ∈ Set.center M) (hz₂ : z₂ ∈ Set.center M) :
    z₁ * z₂ ∈ Set.center M where
  comm a := calc
                                    /-
                                      M : Type u_1
                                      inst✝ : Mul M
                                      z₁ z₂ : M
                                      hz₁ : Membership.mem (Set.center M) z₁
                                      hz₂ : Membership.mem (Set.center M) z₂
                                      a : M
                                      ⊢ Eq (HMul.hMul (HMul.hMul z₁ z₂) a) (HMul.hMul (HMul.hMul z₂ z₁) a)
                                    -/
    z₁ * z₂ * a = z₂ * z₁ * a := by rw [hz₁.comm]
                                    /-
                                      🎉 no goals
                                    -/
                            /-
                              M : Type u_1
                              inst✝ : Mul M
                              z₁ z₂ : M
                              hz₁ : Membership.mem (Set.center M) z₁
                              hz₂ : Membership.mem (Set.center M) z₂
                              a : M
                              ⊢ Eq (HMul.hMul (HMul.hMul z₂ z₁) a) (HMul.hMul z₂ (HMul.hMul z₁ a))
                            -/
    _ = z₂ * (z₁ * a) := by rw [hz₁.mid_assoc z₂]
                            /-
                              🎉 no goals
                            -/
                            /-
                              M : Type u_1
                              inst✝ : Mul M
                              z₁ z₂ : M
                              hz₁ : Membership.mem (Set.center M) z₁
                              hz₂ : Membership.mem (Set.center M) z₂
                              a : M
                              ⊢ Eq (HMul.hMul z₂ (HMul.hMul z₁ a)) (HMul.hMul (HMul.hMul a z₁) z₂)
                            -/
    _ = (a * z₁) * z₂ := by rw [hz₁.comm, hz₂.comm]
                            /-
                              🎉 no goals
                            -/
                            /-
                              M : Type u_1
                              inst✝ : Mul M
                              z₁ z₂ : M
                              hz₁ : Membership.mem (Set.center M) z₁
                              hz₂ : Membership.mem (Set.center M) z₂
                              a : M
                              ⊢ Eq (HMul.hMul (HMul.hMul a z₁) z₂) (HMul.hMul a (HMul.hMul z₁ z₂))
                            -/
    _ = a * (z₁ * z₂) := by rw [hz₂.right_assoc a z₁]
                            /-
                              🎉 no goals
                            -/
  left_assoc (b c : M) := calc
                                                  /-
                                                    M : Type u_1
                                                    inst✝ : Mul M
                                                    z₁ z₂ : M
                                                    hz₁ : Membership.mem (Set.center M) z₁
                                                    hz₂ : Membership.mem (Set.center M) z₂
                                                    b c : M
                                                    ⊢ Eq (HMul.hMul (HMul.hMul z₁ z₂) (HMul.hMul b c)) (HMul.hMul z₁ (HMul.hMul z₂ …
                                                  -/
    z₁ * z₂ * (b * c) = z₁ * (z₂ * (b * c)) := by rw [hz₂.mid_assoc]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                  /-
                                    M : Type u_1
                                    inst✝ : Mul M
                                    z₁ z₂ : M
                                    hz₁ : Membership.mem (Set.center M) z₁
                                    hz₂ : Membership.mem (Set.center M) z₂
                                    b c : M
                                    ⊢ Eq (HMul.hMul z₁ (HMul.hMul z₂ (HMul.hMul b c))) (HMul.hMul z₁ (HMul.hMul (H …
                                  -/
    _ = z₁ * ((z₂ * b) * c) := by rw [hz₂.left_assoc]
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    M : Type u_1
                                    inst✝ : Mul M
                                    z₁ z₂ : M
                                    hz₁ : Membership.mem (Set.center M) z₁
                                    hz₂ : Membership.mem (Set.center M) z₂
                                    b c : M
                                    ⊢ Eq (HMul.hMul z₁ (HMul.hMul (HMul.hMul z₂ b) c)) (HMul.hMul (HMul.hMul z₁ (H …
                                  -/
    _ = (z₁ * (z₂ * b)) * c := by rw [hz₁.left_assoc]
                                  /-
                                    🎉 no goals
                                  -/
                              /-
                                M : Type u_1
                                inst✝ : Mul M
                                z₁ z₂ : M
                                hz₁ : Membership.mem (Set.center M) z₁
                                hz₂ : Membership.mem (Set.center M) z₂
                                b c : M
                                ⊢ Eq (HMul.hMul (HMul.hMul z₁ (HMul.hMul z₂ b)) c) (HMul.hMul (HMul.hMul (HMul …
                              -/
    _ = z₁ * z₂ * b * c := by rw [hz₂.mid_assoc]
                              /-
                                🎉 no goals
                              -/
  mid_assoc (a c : M) := calc
                                                  /-
                                                    M : Type u_1
                                                    inst✝ : Mul M
                                                    z₁ z₂ : M
                                                    hz₁ : Membership.mem (Set.center M) z₁
                                                    hz₂ : Membership.mem (Set.center M) z₂
                                                    a c : M
                                                    ⊢ Eq (HMul.hMul (HMul.hMul a (HMul.hMul z₁ z₂)) c) (HMul.hMul (HMul.hMul (HMul …
                                                  -/
    a * (z₁ * z₂) * c = ((a * z₁) * z₂) * c := by rw [hz₁.mid_assoc]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                  /-
                                    M : Type u_1
                                    inst✝ : Mul M
                                    z₁ z₂ : M
                                    hz₁ : Membership.mem (Set.center M) z₁
                                    hz₂ : Membership.mem (Set.center M) z₂
                                    a c : M
                                    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul a z₁) z₂) c) (HMul.hMul (HMul.hMul a z₁) …
                                  -/
    _ = (a * z₁) * (z₂ * c) := by rw [hz₂.mid_assoc]
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    M : Type u_1
                                    inst✝ : Mul M
                                    z₁ z₂ : M
                                    hz₁ : Membership.mem (Set.center M) z₁
                                    hz₂ : Membership.mem (Set.center M) z₂
                                    a c : M
                                    ⊢ Eq (HMul.hMul (HMul.hMul a z₁) (HMul.hMul z₂ c)) (HMul.hMul a (HMul.hMul z₁  …
                                  -/
    _ = a * (z₁ * (z₂ * c)) := by rw [hz₁.mid_assoc]
                                  /-
                                    🎉 no goals
                                  -/
                                /-
                                  M : Type u_1
                                  inst✝ : Mul M
                                  z₁ z₂ : M
                                  hz₁ : Membership.mem (Set.center M) z₁
                                  hz₂ : Membership.mem (Set.center M) z₂
                                  a c : M
                                  ⊢ Eq (HMul.hMul a (HMul.hMul z₁ (HMul.hMul z₂ c))) (HMul.hMul a (HMul.hMul (HM …
                                -/
    _ = a * (z₁ * z₂ * c) := by rw [hz₂.mid_assoc]
                                /-
                                  🎉 no goals
                                -/
  right_assoc (a b : M) := calc
                                                  /-
                                                    M : Type u_1
                                                    inst✝ : Mul M
                                                    z₁ z₂ : M
                                                    hz₁ : Membership.mem (Set.center M) z₁
                                                    hz₂ : Membership.mem (Set.center M) z₂
                                                    a b : M
                                                    ⊢ Eq (HMul.hMul (HMul.hMul a b) (HMul.hMul z₁ z₂)) (HMul.hMul (HMul.hMul (HMul …
                                                  -/
    a * b * (z₁ * z₂) = ((a * b) * z₁) * z₂ := by rw [hz₂.right_assoc]
                                                  /-
                                                    🎉 no goals
                                                  -/
                                  /-
                                    M : Type u_1
                                    inst✝ : Mul M
                                    z₁ z₂ : M
                                    hz₁ : Membership.mem (Set.center M) z₁
                                    hz₂ : Membership.mem (Set.center M) z₂
                                    a b : M
                                    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul a b) z₁) z₂) (HMul.hMul (HMul.hMul a (HM …
                                  -/
    _ = (a * (b * z₁)) * z₂ := by rw [hz₁.right_assoc]
                                  /-
                                    🎉 no goals
                                  -/
                                   /-
                                     M : Type u_1
                                     inst✝ : Mul M
                                     z₁ z₂ : M
                                     hz₁ : Membership.mem (Set.center M) z₁
                                     hz₂ : Membership.mem (Set.center M) z₂
                                     a b : M
                                     ⊢ Eq (HMul.hMul (HMul.hMul a (HMul.hMul b z₁)) z₂) (HMul.hMul a (HMul.hMul (HM …
                                   -/
    _ =  a * ((b * z₁) * z₂) := by rw [hz₂.right_assoc]
                                   /-
                                     🎉 no goals
                                   -/
                                  /-
                                    M : Type u_1
                                    inst✝ : Mul M
                                    z₁ z₂ : M
                                    hz₁ : Membership.mem (Set.center M) z₁
                                    hz₂ : Membership.mem (Set.center M) z₂
                                    a b : M
                                    ⊢ Eq (HMul.hMul a (HMul.hMul (HMul.hMul b z₁) z₂)) (HMul.hMul a (HMul.hMul b ( …
                                  -/
    _ = a * (b * (z₁ * z₂)) := by rw [hz₁.mid_assoc]
                                  /-
                                    🎉 no goals
                                  -/


@[to_additive addCenter_subset_addCentralizer]
lemma center_subset_centralizer (S : Set M) : Set.center M ⊆ S.centralizer :=
  fun _ hx m _ ↦ (hx.comm m).symm


@[to_additive addCentralizer_union]
lemma centralizer_union : centralizer (S ∪ T) = centralizer S ∩ centralizer T := by
  /-
    M : Type u_1
    S T : Set M
    inst✝ : Mul M
    ⊢ Eq (Union.union S T).centralizer (Inter.inter S.centralizer T.centralizer)
  -/
  simp [centralizer, or_imp, forall_and, setOf_and]
  /-
    🎉 no goals
  -/


@[to_additive (attr := gcongr) addCentralizer_subset]
lemma centralizer_subset (h : S ⊆ T) : centralizer T ⊆ centralizer S := fun _ ht s hs ↦ ht s (h hs)


@[to_additive subset_addCentralizer_addCentralizer]
lemma subset_centralizer_centralizer : S ⊆ S.centralizer.centralizer := by
  /-
    M : Type u_1
    S : Set M
    inst✝ : Mul M
    ⊢ HasSubset.Subset S S.centralizer.centralizer
  -/
  intro x hx
  /-
    M : Type u_1
    S : Set M
    inst✝ : Mul M
    x : M
    hx : Membership.mem S x
    ⊢ Membership.mem S.centralizer.centralizer x
  -/
  simp only [Set.mem_centralizer_iff]
  /-
    M : Type u_1
    S : Set M
    inst✝ : Mul M
    x : M
    hx : Membership.mem S x
    ⊢ ∀ (m : M), (∀ (m_1 : M), Membership.mem S m_1 → Eq (HMul.hMul m_1 m) (HMul.h …
  -/
  exact fun y hy => (hy x hx).symm
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) addCentralizer_addCentralizer_addCentralizer]
lemma centralizer_centralizer_centralizer (S : Set M) :
    S.centralizer.centralizer.centralizer = S.centralizer := by
  /-
    M : Type u_1
    inst✝ : Mul M
    S : Set M
    ⊢ Eq S.centralizer.centralizer.centralizer S.centralizer
  -/
  refine Set.Subset.antisymm ?_ Set.subset_centralizer_centralizer
  /-
    M : Type u_1
    inst✝ : Mul M
    S : Set M
    ⊢ HasSubset.Subset S.centralizer.centralizer.centralizer S.centralizer
  -/
  intro x hx
  /-
    M : Type u_1
    inst✝ : Mul M
    S : Set M
    x : M
    hx : Membership.mem S.centralizer.centralizer.centralizer x
    ⊢ Membership.mem S.centralizer x
  -/
  rw [Set.mem_centralizer_iff]
  /-
    M : Type u_1
    inst✝ : Mul M
    S : Set M
    x : M
    hx : Membership.mem S.centralizer.centralizer.centralizer x
    ⊢ ∀ (m : M), Membership.mem S m → Eq (HMul.hMul m x) (HMul.hMul x m)
  -/
  intro y hy
  /-
    M : Type u_1
    inst✝ : Mul M
    S : Set M
    x : M
    hx : Membership.mem S.centralizer.centralizer.centralizer x
    y : M
    hy : Membership.mem S y
    ⊢ Eq (HMul.hMul y x) (HMul.hMul x y)
  -/
  rw [Set.mem_centralizer_iff] at hx
  /-
    M : Type u_1
    inst✝ : Mul M
    S : Set M
    x : M
    hx : ∀ (m : M), Membership.mem S.centralizer.centralizer m → Eq (HMul.hMul m x …
    y : M
    hy : Membership.mem S y
    ⊢ Eq (HMul.hMul y x) (HMul.hMul x y)
  -/
  exact hx y <| Set.subset_centralizer_centralizer hy
  /-
    🎉 no goals
  -/


@[to_additive decidableMemAddCentralizer]
instance decidableMemCentralizer [∀ a : M, Decidable <| ∀ b ∈ S, b * a = a * b] :
    DecidablePred (· ∈ centralizer S) := fun _ ↦ decidable_of_iff' _ mem_centralizer_iff


@[to_additive addCentralizer_addCentralizer_comm_of_comm]
lemma centralizer_centralizer_comm_of_comm (h_comm : ∀ x ∈ S, ∀ y ∈ S, x * y = y * x) :
    ∀ x ∈ S.centralizer.centralizer, ∀ y ∈ S.centralizer.centralizer, x * y = y * x :=
  fun _ h₁ _ h₂ ↦ h₂ _ fun _ h₃ ↦ h₁ _ fun _ h₄ ↦ h_comm _ h₄ _ h₃


@[to_additive]
theorem _root_.Semigroup.mem_center_iff {z : M} :
                                                           /-
                                                             M : Type u_1
                                                             inst✝ : Semigroup M
                                                             z : M
                                                             a : Membership.mem (Set.center M) z
                                                             g : M
                                                             ⊢ Eq (HMul.hMul g z) (HMul.hMul z g)
                                                           -/
    z ∈ Set.center M ↔ ∀ g, g * z = z * g := ⟨fun a g ↦ by rw [IsMulCentral.comm a g],
                                                           /-
                                                             🎉 no goals
                                                           -/
  fun h ↦ ⟨fun _ ↦ (Commute.eq (h _)).symm, fun _ _ ↦ (mul_assoc z _ _).symm,
  fun _ _ ↦ mul_assoc _ z _, fun _ _ ↦ mul_assoc _ _ z⟩ ⟩


@[to_additive (attr := simp) add_mem_addCentralizer]
lemma mul_mem_centralizer (ha : a ∈ centralizer S) (hb : b ∈ centralizer S) :
    a * b ∈ centralizer S := fun g hg ↦ by
  /-
    M : Type u_1
    S : Set M
    inst✝ : Semigroup M
    a b : M
    ha : Membership.mem S.centralizer a
    hb : Membership.mem S.centralizer b
    g : M
    hg : Membership.mem S g
    ⊢ Eq (HMul.hMul g (HMul.hMul a b)) (HMul.hMul (HMul.hMul a b) g)
  -/
  rw [mul_assoc, ← hb g hg, ← mul_assoc, ha g hg, mul_assoc]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) addCentralizer_eq_top_iff_subset]
theorem centralizer_eq_top_iff_subset : centralizer S = Set.univ ↔ S ⊆ center M :=
  eq_top_iff.trans <| ⟨
                                                         /-
                                                           M : Type u_1
                                                           S : Set M
                                                           inst✝ : Semigroup M
                                                           h : LE.le Top.top S.centralizer
                                                           x✝¹ : M
                                                           hx : Membership.mem S x✝¹
                                                           x✝ : M
                                                           ⊢ Eq (HMul.hMul x✝ x✝¹) (HMul.hMul x✝¹ x✝)
                                                         -/
    fun h _ hx ↦ Semigroup.mem_center_iff.mpr fun _ ↦ by rw [h trivial _ hx],
                                                         /-
                                                           🎉 no goals
                                                         -/
    fun h _ _ _ hm ↦ (h hm).comm _⟩


variable (M) in
@[to_additive (attr := simp) addCentralizer_univ]
lemma centralizer_univ : centralizer univ = center M :=
  Subset.antisymm (fun _ ha ↦ Semigroup.mem_center_iff.mpr fun b ↦ ha b (Set.mem_univ b))
  fun _ ha b _ ↦ (ha.comm b).symm

-- TODO Add `instance : Decidable (IsMulCentral a)` for `instance decidableMemCenter [Mul M]`

@[to_additive decidableMemAddCenter]
instance decidableMemCenter [∀ a : M, Decidable <| ∀ b : M, b * a = a * b] :
    DecidablePred (· ∈ center M) := fun _ => decidable_of_iff' _ (Semigroup.mem_center_iff)


@[to_additive (attr := simp) addCenter_eq_univ]
theorem center_eq_univ : center M = univ :=
  (Subset.antisymm (subset_univ _)) fun _ _ => Semigroup.mem_center_iff.mpr (fun _ => mul_comm _ _)


@[to_additive (attr := simp) addCentralizer_eq_univ]
lemma centralizer_eq_univ : centralizer S = univ :=
  eq_univ_of_forall fun _ _ _ ↦ mul_comm _ _


@[to_additive (attr := simp) zero_mem_addCenter]
theorem one_mem_center : (1 : M) ∈ Set.center M where
                /-
                  M : Type u_1
                  inst✝ : MulOneClass M
                  x✝ : M
                  ⊢ Eq (HMul.hMul 1 x✝) (HMul.hMul x✝ 1)
                -/
  comm _  := by rw [one_mul, mul_one]
                /-
                  🎉 no goals
                -/
                       /-
                         M : Type u_1
                         inst✝ : MulOneClass M
                         x✝¹ x✝ : M
                         ⊢ Eq (HMul.hMul 1 (HMul.hMul x✝¹ x✝)) (HMul.hMul (HMul.hMul 1 x✝¹) x✝)
                       -/
  left_assoc _ _ := by rw [one_mul, one_mul]
                       /-
                         🎉 no goals
                       -/
                      /-
                        M : Type u_1
                        inst✝ : MulOneClass M
                        x✝¹ x✝ : M
                        ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ 1) x✝) (HMul.hMul x✝¹ (HMul.hMul 1 x✝))
                      -/
  mid_assoc _ _ := by rw [mul_one, one_mul]
                      /-
                        🎉 no goals
                      -/
                        /-
                          M : Type u_1
                          inst✝ : MulOneClass M
                          x✝¹ x✝ : M
                          ⊢ Eq (HMul.hMul (HMul.hMul x✝¹ x✝) 1) (HMul.hMul x✝¹ (HMul.hMul x✝ 1))
                        -/
  right_assoc _ _ := by rw [mul_one, mul_one]
                        /-
                          🎉 no goals
                        -/


@[to_additive (attr := simp) zero_mem_addCentralizer]
                                                          /-
                                                            M : Type u_1
                                                            S : Set M
                                                            inst✝ : MulOneClass M
                                                            ⊢ Membership.mem S.centralizer 1
                                                          -/
lemma one_mem_centralizer : (1 : M) ∈ centralizer S := by simp [mem_centralizer_iff]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[to_additive subset_addCenter_add_units]
theorem subset_center_units : ((↑) : Mˣ → M) ⁻¹' center M ⊆ Set.center Mˣ :=
  fun _ ha => by
  /-
    M : Type u_1
    inst✝ : Monoid M
    x✝ : Units M
    ha : Membership.mem (Set.preimage Units.val (Set.center M)) x✝
    ⊢ Membership.mem (Set.center (Units M)) x✝
  -/
  rw [_root_.Semigroup.mem_center_iff]
  /-
    M : Type u_1
    inst✝ : Monoid M
    x✝ : Units M
    ha : Membership.mem (Set.preimage Units.val (Set.center M)) x✝
    ⊢ ∀ (g : Units M), Eq (HMul.hMul g x✝) (HMul.hMul x✝ g)
  -/
  intro _
  /-
    M : Type u_1
    inst✝ : Monoid M
    x✝ : Units M
    ha : Membership.mem (Set.preimage Units.val (Set.center M)) x✝
    g✝ : Units M
    ⊢ Eq (HMul.hMul g✝ x✝) (HMul.hMul x✝ g✝)
  -/
  rw [← Units.eq_iff, Units.val_mul, Units.val_mul, ha.comm]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem units_inv_mem_center {a : Mˣ} (ha : ↑a ∈ Set.center M) : ↑a⁻¹ ∈ Set.center M := by
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Units M
    ha : Membership.mem (Set.center M) ↑a
    ⊢ Membership.mem (Set.center M) ↑(Inv.inv a)
  -/
  rw [Semigroup.mem_center_iff] at *
  /-
    M : Type u_1
    inst✝ : Monoid M
    a : Units M
    ha : ∀ (g : M), Eq (HMul.hMul g ↑a) (HMul.hMul (↑a) g)
    ⊢ ∀ (g : M), Eq (HMul.hMul g ↑(Inv.inv a)) (HMul.hMul (↑(Inv.inv a)) g)
  -/
  exact (Commute.units_inv_right <| ha ·)
  /-
    🎉 no goals
  -/


@[simp]
theorem invOf_mem_center {a : M} [Invertible a] (ha : a ∈ Set.center M) : ⅟a ∈ Set.center M := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    a : M
    inst✝ : Invertible a
    ha : Membership.mem (Set.center M) a
    ⊢ Membership.mem (Set.center M) (Invertible.invOf a)
  -/
  rw [Semigroup.mem_center_iff] at *
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    a : M
    inst✝ : Invertible a
    ha : ∀ (g : M), Eq (HMul.hMul g a) (HMul.hMul a g)
    ⊢ ∀ (g : M), Eq (HMul.hMul g (Invertible.invOf a)) (HMul.hMul (Invertible.invO …
  -/
  exact (Commute.invOf_right <| ha ·)
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) neg_mem_addCenter]
theorem inv_mem_center (ha : a ∈ Set.center M) : a⁻¹ ∈ Set.center M := by
  /-
    M : Type u_1
    inst✝ : DivisionMonoid M
    a : M
    ha : Membership.mem (Set.center M) a
    ⊢ Membership.mem (Set.center M) (Inv.inv a)
  -/
  rw [_root_.Semigroup.mem_center_iff]
  /-
    M : Type u_1
    inst✝ : DivisionMonoid M
    a : M
    ha : Membership.mem (Set.center M) a
    ⊢ ∀ (g : M), Eq (HMul.hMul g (Inv.inv a)) (HMul.hMul (Inv.inv a) g)
  -/
  intro _
  /-
    M : Type u_1
    inst✝ : DivisionMonoid M
    a : M
    ha : Membership.mem (Set.center M) a
    g✝ : M
    ⊢ Eq (HMul.hMul g✝ (Inv.inv a)) (HMul.hMul (Inv.inv a) g✝)
  -/
  rw [← inv_inj, mul_inv_rev, inv_inv, ha.comm, mul_inv_rev, inv_inv]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) sub_mem_addCenter]
theorem div_mem_center (ha : a ∈ Set.center M) (hb : b ∈ Set.center M) : a / b ∈ Set.center M := by
  /-
    M : Type u_1
    inst✝ : DivisionMonoid M
    a b : M
    ha : Membership.mem (Set.center M) a
    hb : Membership.mem (Set.center M) b
    ⊢ Membership.mem (Set.center M) (HDiv.hDiv a b)
  -/
  rw [div_eq_mul_inv]
  /-
    M : Type u_1
    inst✝ : DivisionMonoid M
    a b : M
    ha : Membership.mem (Set.center M) a
    hb : Membership.mem (Set.center M) b
    ⊢ Membership.mem (Set.center M) (HMul.hMul a (Inv.inv b))
  -/
  exact mul_mem_center ha (inv_mem_center hb)
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp) neg_mem_addCentralizer]
lemma inv_mem_centralizer (ha : a ∈ centralizer S) : a⁻¹ ∈ centralizer S :=
                /-
                  M : Type u_1
                  S : Set M
                  inst✝ : Group M
                  a : M
                  ha : Membership.mem S.centralizer a
                  g : M
                  hg : Membership.mem S g
                  ⊢ Eq (HMul.hMul g (Inv.inv a)) (HMul.hMul (Inv.inv a) g)
                -/
  fun g hg ↦ by rw [mul_inv_eq_iff_eq_mul, mul_assoc, eq_inv_mul_iff_mul_eq, ha g hg]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp) sub_mem_addCentralizer]
lemma div_mem_centralizer (ha : a ∈ centralizer S) (hb : b ∈ centralizer S) :
    a / b ∈ centralizer S := by
  /-
    M : Type u_1
    S : Set M
    inst✝ : Group M
    a b : M
    ha : Membership.mem S.centralizer a
    hb : Membership.mem S.centralizer b
    ⊢ Membership.mem S.centralizer (HDiv.hDiv a b)
  -/
  simpa only [div_eq_mul_inv] using mul_mem_centralizer ha (inv_mem_centralizer hb)
  /-
    🎉 no goals
  -/


