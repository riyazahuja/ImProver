/-- The Coxeter relation associated to a Coxeter matrix $M$ and two indices $i, i' \in B$.
That is, the relation $(s_i s_{i'})^{M_{i, i'}}$, considered as an element of the free group
on $\{s_i\}_{i \in B}$.
If $M_{i, i'} = 0$, then this is the identity, indicating that there is no relation between
$s_i$ and $s_{i'}$. -/
def relation (i i' : B) : FreeGroup B := (FreeGroup.of i * FreeGroup.of i') ^ M i i'


/-- The set of all Coxeter relations associated to the Coxeter matrix $M$. -/
def relationsSet : Set (FreeGroup B) := range <| uncurry M.relation


/-- The Coxeter group associated to a Coxeter matrix $M$; that is, the group
$$\langle \{s_i\}_{i \in B} \vert \{(s_i s_{i'})^{M_{i, i'}}\}_{i, i' \in B} \rangle.$$ -/
protected def Group : Type _ := PresentedGroup M.relationsSet


instance : Group M.Group := QuotientGroup.Quotient.group _


/-- The simple reflection of the Coxeter group `M.group` at the index `i`. -/
def simple (i : B) : M.Group := PresentedGroup.of i


theorem reindex_relationsSet :
    (M.reindex e).relationsSet =
    FreeGroup.freeGroupCongr e '' M.relationsSet := let M' := M.reindex e; calc
  Set.range (uncurry M'.relation)
                                                           /-
                                                             B : Type u_1
                                                             B' : Type u_2
                                                             M : CoxeterMatrix B
                                                             e : Equiv B B'
                                                             M' : CoxeterMatrix B' := CoxeterMatrix.reindex e M
                                                             ⊢ Eq (Set.range (Function.uncurry M'.relation)) (Set.range (Function.comp (Fun …
                                                           -/
  _ = Set.range (uncurry M'.relation ∘ Prod.map e e) := by simp [Set.range_comp]
                                                           /-
                                                             🎉 no goals
                                                           -/
  _ = Set.range (FreeGroup.freeGroupCongr e ∘ uncurry M.relation) := by
      /-
        B : Type u_1
        B' : Type u_2
        M : CoxeterMatrix B
        e : Equiv B B'
        M' : CoxeterMatrix B' := CoxeterMatrix.reindex e M
        ⊢ Eq (Set.range (Function.comp (Function.uncurry M'.relation) (Prod.map ⇑e ⇑e) …
      -/
      apply congrArg Set.range
      /-
        B : Type u_1
        B' : Type u_2
        M : CoxeterMatrix B
        e : Equiv B B'
        M' : CoxeterMatrix B' := CoxeterMatrix.reindex e M
        ⊢ Eq (Function.comp (Function.uncurry M'.relation) (Prod.map ⇑e ⇑e)) (Function …
      -/
      ext ⟨i, i'⟩
      /-
        case h.mk
        B : Type u_1
        B' : Type u_2
        M : CoxeterMatrix B
        e : Equiv B B'
        M' : CoxeterMatrix B' := CoxeterMatrix.reindex e M
        i i' : B
        ⊢ Eq (Function.comp (Function.uncurry M'.relation) (Prod.map ⇑e ⇑e) { fst := i …
      -/
      simp [relation, reindex_apply, M']
      /-
        🎉 no goals
      -/
              /-
                B : Type u_1
                B' : Type u_2
                M : CoxeterMatrix B
                e : Equiv B B'
                M' : CoxeterMatrix B' := CoxeterMatrix.reindex e M
                ⊢ Eq (Set.range (Function.comp (⇑(FreeGroup.freeGroupCongr e)) (Function.uncur …
              -/
  _ = _ := by simp [Set.range_comp, relationsSet]
              /-
                🎉 no goals
              -/


/-- The isomorphism between the Coxeter group associated to the reindexed matrix `M.reindex e` and
the Coxeter group associated to `M`. -/
def reindexGroupEquiv : (M.reindex e).Group ≃* M.Group :=
  .symm <| QuotientGroup.congr
    (Subgroup.normalClosure M.relationsSet)
    (Subgroup.normalClosure (M.reindex e).relationsSet)
    (FreeGroup.freeGroupCongr e)
    (by
      rw [reindex_relationsSet,
        Subgroup.map_normalClosure _ _ (by simpa using (FreeGroup.freeGroupCongr e).surjective),
        MonoidHom.coe_coe])


theorem reindexGroupEquiv_apply_simple (i : B') :
    (M.reindexGroupEquiv e) ((M.reindex e).simple i) = M.simple (e.symm i) := rfl


theorem reindexGroupEquiv_symm_apply_simple (i : B) :
    (M.reindexGroupEquiv e).symm (M.simple i) = (M.reindex e).simple (e i) := rfl


/-- A Coxeter system `CoxeterSystem M W` is a structure recording the isomorphism between
a group `W` and the Coxeter group associated to a Coxeter matrix `M`. -/
@[ext]
structure CoxeterSystem (W : Type*) [Group W] where
  /-- The isomorphism between `W` and the Coxeter group associated to `M`. -/
  mulEquiv : W ≃* M.Group


/-- A group is a Coxeter group if it admits a Coxeter system for some Coxeter matrix `M`. -/
class IsCoxeterGroup.{u} (W : Type u) [Group W] : Prop where
  nonempty_system : ∃ B : Type u, ∃ M : CoxeterMatrix B, Nonempty (CoxeterSystem M W)


/-- The canonical Coxeter system on the Coxeter group associated to `M`. -/
def CoxeterMatrix.toCoxeterSystem : CoxeterSystem M M.Group := ⟨.refl _⟩


/-- Reindex a Coxeter system through a bijection of the indexing sets. -/
@[simps]
protected def reindex (e : B ≃ B') : CoxeterSystem (M.reindex e) W :=
  ⟨cs.mulEquiv.trans (M.reindexGroupEquiv e).symm⟩


/-- Push a Coxeter system through a group isomorphism. -/
@[simps]
protected def map (e : W ≃* H) : CoxeterSystem M H := ⟨e.symm.trans cs.mulEquiv⟩


/-- The simple reflection of `W` at the index `i`. -/
def simple (i : B) : W := cs.mulEquiv.symm (PresentedGroup.of i)


@[simp]
theorem _root_.CoxeterMatrix.toCoxeterSystem_simple (M : CoxeterMatrix B) :
    M.toCoxeterSystem.simple = M.simple := rfl


@[simp] theorem reindex_simple (i' : B') : (cs.reindex e).simple i' = cs.simple (e.symm i') := rfl


@[simp] theorem map_simple (e : W ≃* H) (i : B) : (cs.map e).simple i = e (cs.simple i) := rfl


local prefix:100 "s" => cs.simple


@[simp]
theorem simple_mul_simple_self (i : B) : s i * s i = 1 := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    ⊢ Eq (HMul.hMul (cs.simple i) (cs.simple i)) 1
  -/
  have : (FreeGroup.of i) * (FreeGroup.of i) ∈ M.relationsSet := ⟨(i, i), by simp [relation]⟩
  have : (PresentedGroup.mk _ (FreeGroup.of i * FreeGroup.of i) : M.Group) = 1 :=
    (QuotientGroup.eq_one_iff _).mpr (Subgroup.subset_normalClosure this)
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    this✝ : Membership.mem M.relationsSet (HMul.hMul (FreeGroup.of i) (FreeGroup.o …
    this : Eq ((PresentedGroup.mk M.relationsSet) (HMul.hMul (FreeGroup.of i) (Fre …
    ⊢ Eq (HMul.hMul (cs.simple i) (cs.simple i)) 1
  -/
  unfold simple
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    this✝ : Membership.mem M.relationsSet (HMul.hMul (FreeGroup.of i) (FreeGroup.o …
    this : Eq ((PresentedGroup.mk M.relationsSet) (HMul.hMul (FreeGroup.of i) (Fre …
    ⊢ Eq (HMul.hMul (cs.mulEquiv.symm (PresentedGroup.of i)) (cs.mulEquiv.symm (Pr …
  -/
  rw [← map_mul, PresentedGroup.of, map_mul]
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i : B
    this✝ : Membership.mem M.relationsSet (HMul.hMul (FreeGroup.of i) (FreeGroup.o …
    this : Eq ((PresentedGroup.mk M.relationsSet) (HMul.hMul (FreeGroup.of i) (Fre …
    ⊢ Eq (HMul.hMul (cs.mulEquiv.symm ((PresentedGroup.mk M.relationsSet) (FreeGro …
  -/
  exact map_mul_eq_one cs.mulEquiv.symm this
  /-
    🎉 no goals
  -/


@[simp]
theorem simple_mul_simple_cancel_right {w : W} (i : B) : w * s i * s i = w := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Eq (HMul.hMul (HMul.hMul w (cs.simple i)) (cs.simple i)) w
  -/
  simp [mul_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem simple_mul_simple_cancel_left {w : W} (i : B) : s i * (s i * w) = w := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    i : B
    ⊢ Eq (HMul.hMul (cs.simple i) (HMul.hMul (cs.simple i) w)) w
  -/
  simp [← mul_assoc]
  /-
    🎉 no goals
  -/


@[simp] theorem simple_sq (i : B) : s i ^ 2 = 1 := pow_two (s i) ▸ cs.simple_mul_simple_self i


@[simp]
theorem inv_simple (i : B) : (s i)⁻¹ = s i :=
  (eq_inv_of_mul_eq_one_right (cs.simple_mul_simple_self i)).symm


@[simp]
theorem simple_mul_simple_pow (i i' : B) : (s i * s i') ^ M i i' = 1 := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    ⊢ Eq (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple i')) (M.M i i')) 1
  -/
  have : (FreeGroup.of i * FreeGroup.of i') ^ M i i' ∈ M.relationsSet := ⟨(i, i'), rfl⟩
  have : (PresentedGroup.mk _ ((FreeGroup.of i * FreeGroup.of i') ^ M i i') : M.Group) = 1 :=
    (QuotientGroup.eq_one_iff _).mpr (Subgroup.subset_normalClosure this)
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    this✝ : Membership.mem M.relationsSet (HPow.hPow (HMul.hMul (FreeGroup.of i) ( …
    this : Eq ((PresentedGroup.mk M.relationsSet) (HPow.hPow (HMul.hMul (FreeGroup …
    ⊢ Eq (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple i')) (M.M i i')) 1
  -/
  unfold simple
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    this✝ : Membership.mem M.relationsSet (HPow.hPow (HMul.hMul (FreeGroup.of i) ( …
    this : Eq ((PresentedGroup.mk M.relationsSet) (HPow.hPow (HMul.hMul (FreeGroup …
    ⊢ Eq (HPow.hPow (HMul.hMul (cs.mulEquiv.symm (PresentedGroup.of i)) (cs.mulEqu …
  -/
  rw [← map_mul, ← map_pow]
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    this✝ : Membership.mem M.relationsSet (HPow.hPow (HMul.hMul (FreeGroup.of i) ( …
    this : Eq ((PresentedGroup.mk M.relationsSet) (HPow.hPow (HMul.hMul (FreeGroup …
    ⊢ Eq (cs.mulEquiv.symm (HPow.hPow (HMul.hMul (PresentedGroup.of i) (PresentedG …
  -/
  exact (MulEquiv.map_eq_one_iff cs.mulEquiv.symm).mpr this
  /-
    🎉 no goals
  -/


@[simp] theorem simple_mul_simple_pow' (i i' : B) : (s i' * s i) ^ M i i' = 1 :=
  M.symmetric i' i ▸ cs.simple_mul_simple_pow i' i


/-- The simple reflections of `W` generate `W` as a group. -/
theorem subgroup_closure_range_simple : Subgroup.closure (range cs.simple) = ⊤ := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ⊢ Eq (Subgroup.closure (Set.range cs.simple)) Top.top
  -/
  have : cs.simple = cs.mulEquiv.symm ∘ PresentedGroup.of := rfl
  rw [this, Set.range_comp, ← MulEquiv.coe_toMonoidHom, ← MonoidHom.map_closure,
    PresentedGroup.closure_range_of, ← MonoidHom.range_eq_map]
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    this : Eq cs.simple (Function.comp (⇑cs.mulEquiv.symm) PresentedGroup.of)
    ⊢ Eq cs.mulEquiv.symm.toMonoidHom.range Top.top
  -/
  exact MonoidHom.range_eq_top.2 (MulEquiv.surjective _)
  /-
    🎉 no goals
  -/


/-- The simple reflections of `W` generate `W` as a monoid. -/
theorem submonoid_closure_range_simple : Submonoid.closure (range cs.simple) = ⊤ := by
  have : range cs.simple = range cs.simple ∪ (range cs.simple)⁻¹ := by
    simp_rw [inv_range, inv_simple, union_self]
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    this : Eq (Set.range cs.simple) (Union.union (Set.range cs.simple) (Inv.inv (S …
    ⊢ Eq (Submonoid.closure (Set.range cs.simple)) Top.top
  -/
  rw [this, ← Subgroup.closure_toSubmonoid, subgroup_closure_range_simple, Subgroup.top_toSubmonoid]
  /-
    🎉 no goals
  -/


/-- If `p : W → Prop` holds for all simple reflections, it holds for the identity, and it is
preserved under multiplication, then it holds for all elements of `W`. -/
theorem simple_induction {p : W → Prop} (w : W) (simple : ∀ i : B, p (s i)) (one : p 1)
    (mul : ∀ w w' : W, p w → p w' → p (w * w')) : p w := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    p : W → Prop
    w : W
    simple : ∀ (i : B), p (cs.simple i)
    one : p 1
    mul : ∀ (w w' : W), p w → p w' → p (HMul.hMul w w')
    ⊢ p w
  -/
  have := cs.submonoid_closure_range_simple.symm ▸ Submonoid.mem_top w
  exact Submonoid.closure_induction (fun x ⟨i, hi⟩ ↦ hi ▸ simple i) one (fun _ _ _ _ ↦ mul _ _)
    this


/-- If `p : W → Prop` holds for the identity and it is preserved under multiplying on the left
by a simple reflection, then it holds for all elements of `W`. -/
theorem simple_induction_left {p : W → Prop} (w : W) (one : p 1)
    (mul_simple_left : ∀ (w : W) (i : B), p w → p (s i * w)) : p w := by
  let p' : (w : W) → w ∈ Submonoid.closure (Set.range cs.simple) → Prop :=
    fun w _ ↦ p w
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    p : W → Prop
    w : W
    one : p 1
    mul_simple_left : ∀ (w : W) (i : B), p w → p (HMul.hMul (cs.simple i) w)
    p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
    ⊢ p w
  -/
  have := cs.submonoid_closure_range_simple.symm ▸ Submonoid.mem_top w
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    p : W → Prop
    w : W
    one : p 1
    mul_simple_left : ∀ (w : W) (i : B), p w → p (HMul.hMul (cs.simple i) w)
    p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
    this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
    ⊢ p w
  -/
  apply Submonoid.closure_induction_left (p := p')
    /-
      case one
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_left : ∀ (w : W) (i : B), p w → p (HMul.hMul (cs.simple i) w)
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      ⊢ p' 1 ⋯
    -/
  · exact one
    /-
      🎉 no goals
    -/
    /-
      case mul_left
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_left : ∀ (w : W) (i : B), p w → p (HMul.hMul (cs.simple i) w)
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      ⊢ ∀ (x : W) (hx : Membership.mem (Set.range cs.simple) x) (y : W) (hy : Member …
    -/
  · rintro _ ⟨i, rfl⟩ y _
    /-
      case mul_left.intro
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_left : ∀ (w : W) (i : B), p w → p (HMul.hMul (cs.simple i) w)
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      i : B
      y : W
      hy✝ : Membership.mem (Submonoid.closure (Set.range cs.simple)) y
      ⊢ p' y hy✝ → p' (HMul.hMul (cs.simple i) y) ⋯
    -/
    exact mul_simple_left y i
    /-
      🎉 no goals
    -/
    /-
      case h
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_left : ∀ (w : W) (i : B), p w → p (HMul.hMul (cs.simple i) w)
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      ⊢ Membership.mem (Submonoid.closure (Set.range cs.simple)) w
    -/
  · exact this
    /-
      🎉 no goals
    -/


/-- If `p : W → Prop` holds for the identity and it is preserved under multiplying on the right
by a simple reflection, then it holds for all elements of `W`. -/
theorem simple_induction_right {p : W → Prop} (w : W) (one : p 1)
    (mul_simple_right : ∀ (w : W) (i : B), p w → p (w * s i)) : p w := by
  let p' : ((w : W) → w ∈ Submonoid.closure (Set.range cs.simple) → Prop) :=
    fun w _ ↦ p w
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    p : W → Prop
    w : W
    one : p 1
    mul_simple_right : ∀ (w : W) (i : B), p w → p (HMul.hMul w (cs.simple i))
    p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
    ⊢ p w
  -/
  have := cs.submonoid_closure_range_simple.symm ▸ Submonoid.mem_top w
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    p : W → Prop
    w : W
    one : p 1
    mul_simple_right : ∀ (w : W) (i : B), p w → p (HMul.hMul w (cs.simple i))
    p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
    this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
    ⊢ p w
  -/
  apply Submonoid.closure_induction_right (p := p')
    /-
      case one
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_right : ∀ (w : W) (i : B), p w → p (HMul.hMul w (cs.simple i))
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      ⊢ p' 1 ⋯
    -/
  · exact one
    /-
      🎉 no goals
    -/
    /-
      case mul_right
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_right : ∀ (w : W) (i : B), p w → p (HMul.hMul w (cs.simple i))
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      ⊢ ∀ (x : W) (hx : Membership.mem (Submonoid.closure (Set.range cs.simple)) x)  …
    -/
  · rintro x _ _ ⟨i, rfl⟩
    /-
      case mul_right.intro
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_right : ∀ (w : W) (i : B), p w → p (HMul.hMul w (cs.simple i))
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      x : W
      hx✝ : Membership.mem (Submonoid.closure (Set.range cs.simple)) x
      i : B
      ⊢ p' x hx✝ → p' (HMul.hMul x (cs.simple i)) ⋯
    -/
    exact mul_simple_right x i
    /-
      🎉 no goals
    -/
    /-
      case h
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      p : W → Prop
      w : W
      one : p 1
      mul_simple_right : ∀ (w : W) (i : B), p w → p (HMul.hMul w (cs.simple i))
      p' : (w : W) → Membership.mem (Submonoid.closure (Set.range cs.simple)) w → Pr …
      this : Membership.mem (Submonoid.closure (Set.range cs.simple)) w
      ⊢ Membership.mem (Submonoid.closure (Set.range cs.simple)) w
    -/
  · exact this
    /-
      🎉 no goals
    -/


/-- If two homomorphisms with domain `W` agree on all simple reflections, then they are equal. -/
theorem ext_simple {G : Type*} [Monoid G] {φ₁ φ₂ : W →* G} (h : ∀ i : B, φ₁ (s i) = φ₂ (s i)) :
    φ₁ = φ₂ :=
  MonoidHom.eq_of_eqOn_denseM cs.submonoid_closure_range_simple (fun _ ⟨i, hi⟩ ↦ hi ▸ h i)


/-- The proposition that the values of the function `f : B → G` satisfy the Coxeter relations
corresponding to the matrix `M`. -/
def _root_.CoxeterMatrix.IsLiftable {G : Type*} [Monoid G] (M : CoxeterMatrix B) (f : B → G) :
    Prop := ∀ i i', (f i * f i') ^ M i i' = 1


private theorem relations_liftable {G : Type*} [Group G] {f : B → G} (hf : IsLiftable M f)
    (r : FreeGroup B) (hr : r ∈ M.relationsSet) : (FreeGroup.lift f) r = 1 := by
  /-
    B : Type u_1
    M : CoxeterMatrix B
    G : Type u_5
    inst✝ : Group G
    f : B → G
    hf : M.IsLiftable f
    r : FreeGroup B
    hr : Membership.mem M.relationsSet r
    ⊢ Eq ((FreeGroup.lift f) r) 1
  -/
  rcases hr with ⟨⟨i, i'⟩, rfl⟩
  /-
    case intro.mk
    B : Type u_1
    M : CoxeterMatrix B
    G : Type u_5
    inst✝ : Group G
    f : B → G
    hf : M.IsLiftable f
    i i' : B
    ⊢ Eq ((FreeGroup.lift f) (Function.uncurry M.relation { fst := i, snd := i' }) …
  -/
  rw [uncurry, relation, map_pow, _root_.map_mul, FreeGroup.lift.of, FreeGroup.lift.of]
  /-
    case intro.mk
    B : Type u_1
    M : CoxeterMatrix B
    G : Type u_5
    inst✝ : Group G
    f : B → G
    hf : M.IsLiftable f
    i i' : B
    ⊢ Eq (HPow.hPow (HMul.hMul (f { fst := i, snd := i' }.1) (f { fst := i, snd := …
  -/
  exact hf i i'
  /-
    🎉 no goals
  -/


private def groupLift {G : Type*} [Group G] {f : B → G} (hf : IsLiftable M f) : W →* G :=
  (PresentedGroup.toGroup (relations_liftable hf)).comp cs.mulEquiv.toMonoidHom


private def restrictUnit {G : Type*} [Monoid G] {f : B → G} (hf : IsLiftable M f) (i : B) :
    Gˣ where
  val := f i
  inv := f i
  val_inv := pow_one (f i * f i) ▸ M.diagonal i ▸ hf i i
  inv_val := pow_one (f i * f i) ▸ M.diagonal i ▸ hf i i


private theorem toMonoidHom_apply_symm_apply (a : PresentedGroup (M.relationsSet)) :
    (MulEquiv.toMonoidHom cs.mulEquiv : W →* PresentedGroup (M.relationsSet))
    ((MulEquiv.symm cs.mulEquiv) a) = a := calc
                                                        /-
                                                          B : Type u_1
                                                          W : Type u_3
                                                          inst✝ : Group W
                                                          M : CoxeterMatrix B
                                                          cs : CoxeterSystem M W
                                                          a : PresentedGroup M.relationsSet
                                                          ⊢ Eq (cs.mulEquiv.toMonoidHom (cs.mulEquiv.symm a)) (cs.mulEquiv (cs.mulEquiv. …
                                                        -/
  _ = cs.mulEquiv ((MulEquiv.symm cs.mulEquiv) a) := by rfl
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          B : Type u_1
                                                          W : Type u_3
                                                          inst✝ : Group W
                                                          M : CoxeterMatrix B
                                                          cs : CoxeterSystem M W
                                                          a : PresentedGroup M.relationsSet
                                                          ⊢ Eq (cs.mulEquiv (cs.mulEquiv.symm a)) a
                                                        -/
  _ = _                                           := by rw [MulEquiv.apply_symm_apply]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The universal mapping property of Coxeter systems. For any monoid `G`,
functions `f : B → G` whose values satisfy the Coxeter relations are equivalent to
monoid homomorphisms `f' : W → G`. -/
def lift {G : Type*} [Monoid G] : {f : B → G // IsLiftable M f} ≃ (W →* G) where
  toFun f := MonoidHom.comp (Units.coeHom G) (cs.groupLift
    (show ∀ i i', ((restrictUnit f.property) i * (restrictUnit f.property) i') ^ M i i' = 1 from
      fun i i' ↦ Units.ext (f.property i i')))
  invFun ι := ⟨ι ∘ cs.simple, fun i i' ↦ by
    /-
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      ι : MonoidHom W G
      i i' : B
      ⊢ Eq (HPow.hPow (HMul.hMul (Function.comp (⇑ι) cs.simple i) (Function.comp (⇑ι …
    -/
    rw [comp_apply, comp_apply, ← map_mul, ← map_pow, simple_mul_simple_pow, map_one]⟩
    /-
      🎉 no goals
    -/
  left_inv f := by
    /-
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      f : Subtype fun f => M.IsLiftable f
      ⊢ Eq ((fun ι => ⟨Function.comp (⇑ι) cs.simple, ⋯⟩) ((fun f => (Units.coeHom G) …
    -/
    ext i
    /-
      case a.h
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      f : Subtype fun f => M.IsLiftable f
      i : B
      ⊢ Eq (↑((fun ι => ⟨Function.comp (⇑ι) cs.simple, ⋯⟩) ((fun f => (Units.coeHom  …
    -/
    simp only [MonoidHom.comp_apply, comp_apply, mem_setOf_eq, groupLift, simple]
    rw [← MonoidHom.toFun_eq_coe, toMonoidHom_apply_symm_apply, PresentedGroup.toGroup.of,
      OneHom.toFun_eq_coe, MonoidHom.toOneHom_coe, Units.coeHom_apply, restrictUnit]
  right_inv ι := by
    /-
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      ι : MonoidHom W G
      ⊢ Eq ((fun f => (Units.coeHom G).comp (CoxeterSystem.groupLift cs ⋯)) ((fun ι  …
    -/
    apply cs.ext_simple
    /-
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      ι : MonoidHom W G
      ⊢ ∀ (i : B), Eq (((fun f => (Units.coeHom G).comp (CoxeterSystem.groupLift cs  …
    -/
    intro i
    /-
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      ι : MonoidHom W G
      i : B
      ⊢ Eq (((fun f => (Units.coeHom G).comp (CoxeterSystem.groupLift cs ⋯)) ((fun ι …
    -/
    dsimp only
    rw [groupLift, simple, MonoidHom.comp_apply, MonoidHom.comp_apply, toMonoidHom_apply_symm_apply,
      PresentedGroup.toGroup.of, CoxeterSystem.restrictUnit, Units.coeHom_apply]
    /-
      B : Type u_1
      B' : Type u_2
      e : Equiv B B'
      W : Type u_3
      H : Type u_4
      inst✝² : Group W
      inst✝¹ : Group H
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      G : Type u_5
      inst✝ : Monoid G
      ι : MonoidHom W G
      i : B
      ⊢ Eq (↑{ val := Function.comp (⇑ι) cs.simple i, inv := Function.comp (⇑ι) cs.s …
    -/
    simp only [comp_apply, simple]
    /-
      🎉 no goals
    -/


@[simp]
theorem lift_apply_simple {G : Type*} [Monoid G] {f : B → G} (hf : IsLiftable M f) (i : B) :
    cs.lift ⟨f, hf⟩ (s i) = f i := congrFun (congrArg Subtype.val (cs.lift.left_inv ⟨f, hf⟩)) i


/-- If two Coxeter systems on the same group `W` have the same Coxeter matrix `M : Matrix B B ℕ`
and the same simple reflection map `B → W`, then they are identical. -/
theorem simple_determines_coxeterSystem :
    Injective (simple : CoxeterSystem M W → B → W) := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    ⊢ Function.Injective CoxeterSystem.simple
  -/
  intro cs1 cs2 h
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs1 cs2 : CoxeterSystem M W
    h : Eq cs1.simple cs2.simple
    ⊢ Eq cs1 cs2
  -/
  apply CoxeterSystem.ext
  /-
    case mulEquiv
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs1 cs2 : CoxeterSystem M W
    h : Eq cs1.simple cs2.simple
    ⊢ Eq cs1.mulEquiv cs2.mulEquiv
  -/
  apply MulEquiv.toMonoidHom_injective
  /-
    case mulEquiv.a
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs1 cs2 : CoxeterSystem M W
    h : Eq cs1.simple cs2.simple
    ⊢ Eq cs1.mulEquiv.toMonoidHom cs2.mulEquiv.toMonoidHom
  -/
  apply cs1.ext_simple
  /-
    case mulEquiv.a
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs1 cs2 : CoxeterSystem M W
    h : Eq cs1.simple cs2.simple
    ⊢ ∀ (i : B), Eq (cs1.mulEquiv.toMonoidHom (cs1.simple i)) (cs2.mulEquiv.toMono …
  -/
  intro i
  /-
    case mulEquiv.a
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs1 cs2 : CoxeterSystem M W
    h : Eq cs1.simple cs2.simple
    i : B
    ⊢ Eq (cs1.mulEquiv.toMonoidHom (cs1.simple i)) (cs2.mulEquiv.toMonoidHom (cs1. …
  -/
  nth_rw 2 [h]
  /-
    case mulEquiv.a
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs1 cs2 : CoxeterSystem M W
    h : Eq cs1.simple cs2.simple
    i : B
    ⊢ Eq (cs1.mulEquiv.toMonoidHom (cs1.simple i)) (cs2.mulEquiv.toMonoidHom (cs2. …
  -/
  simp [simple]
  /-
    🎉 no goals
  -/


/-- The product of the simple reflections of `W` corresponding to the indices in `ω`. -/
def wordProd (ω : List B) : W := prod (map cs.simple ω)


local prefix:100 "π" => cs.wordProd


                                              /-
                                                B : Type u_1
                                                W : Type u_3
                                                inst✝ : Group W
                                                M : CoxeterMatrix B
                                                cs : CoxeterSystem M W
                                                ⊢ Eq (cs.wordProd List.nil) 1
                                              -/
@[simp] theorem wordProd_nil : π [] = 1 := by simp [wordProd]
                                              /-
                                                🎉 no goals
                                              -/


                                                                          /-
                                                                            B : Type u_1
                                                                            W : Type u_3
                                                                            inst✝ : Group W
                                                                            M : CoxeterMatrix B
                                                                            cs : CoxeterSystem M W
                                                                            i : B
                                                                            ω : List B
                                                                            ⊢ Eq (cs.wordProd (List.cons i ω)) (HMul.hMul (cs.simple i) (cs.wordProd ω))
                                                                          -/
theorem wordProd_cons (i : B) (ω : List B) : π (i :: ω) = s i * π ω := by simp [wordProd]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                                                 /-
                                                                   B : Type u_1
                                                                   W : Type u_3
                                                                   inst✝ : Group W
                                                                   M : CoxeterMatrix B
                                                                   cs : CoxeterSystem M W
                                                                   i : B
                                                                   ⊢ Eq (cs.wordProd (List.cons i List.nil)) (cs.simple i)
                                                                 -/
@[simp] theorem wordProd_singleton (i : B) : π ([i]) = s i := by simp [wordProd]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


                                                                                /-
                                                                                  B : Type u_1
                                                                                  W : Type u_3
                                                                                  inst✝ : Group W
                                                                                  M : CoxeterMatrix B
                                                                                  cs : CoxeterSystem M W
                                                                                  i : B
                                                                                  ω : List B
                                                                                  ⊢ Eq (cs.wordProd (ω.concat i)) (HMul.hMul (cs.wordProd ω) (cs.simple i))
                                                                                -/
theorem wordProd_concat (i : B) (ω : List B) : π (ω.concat i) = π ω * s i := by simp [wordProd]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


                                                                         /-
                                                                           B : Type u_1
                                                                           W : Type u_3
                                                                           inst✝ : Group W
                                                                           M : CoxeterMatrix B
                                                                           cs : CoxeterSystem M W
                                                                           ω ω' : List B
                                                                           ⊢ Eq (cs.wordProd (HAppend.hAppend ω ω')) (HMul.hMul (cs.wordProd ω) (cs.wordP …
                                                                         -/
theorem wordProd_append (ω ω' : List B) : π (ω ++ ω') = π ω * π ω' := by simp [wordProd]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp] theorem wordProd_reverse (ω : List B) : π (reverse ω) = (π ω)⁻¹ := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ω : List B
    ⊢ Eq (cs.wordProd ω.reverse) (Inv.inv (cs.wordProd ω))
  -/
  induction' ω with x ω' ih
    /-
      case nil
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      ⊢ Eq (cs.wordProd List.nil.reverse) (Inv.inv (cs.wordProd List.nil))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      x : B
      ω' : List B
      ih : Eq (cs.wordProd ω'.reverse) (Inv.inv (cs.wordProd ω'))
      ⊢ Eq (cs.wordProd (List.cons x ω').reverse) (Inv.inv (cs.wordProd (List.cons x …
    -/
  · simpa [wordProd_cons, wordProd_append] using ih
    /-
      🎉 no goals
    -/


theorem wordProd_surjective : Surjective cs.wordProd := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    ⊢ Function.Surjective cs.wordProd
  -/
  intro w
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    w : W
    ⊢ Exists fun a => Eq (cs.wordProd a) w
  -/
  apply cs.simple_induction_left w
    /-
      case one
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ Exists fun a => Eq (cs.wordProd a) 1
    -/
  · use []
    /-
      case h
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ Eq (cs.wordProd List.nil) 1
    -/
    rw [wordProd_nil]
    /-
      🎉 no goals
    -/
    /-
      case mul_simple_left
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      ⊢ ∀ (w : W) (i : B), (Exists fun a => Eq (cs.wordProd a) w) → Exists fun a =>  …
    -/
  · rintro _ i ⟨ω, rfl⟩
    /-
      case mul_simple_left.intro
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ω : List B
      ⊢ Exists fun a => Eq (cs.wordProd a) (HMul.hMul (cs.simple i) (cs.wordProd ω))
    -/
    use i :: ω
    /-
      case h
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      w : W
      i : B
      ω : List B
      ⊢ Eq (cs.wordProd (List.cons i ω)) (HMul.hMul (cs.simple i) (cs.wordProd ω))
    -/
    rw [wordProd_cons]
    /-
      🎉 no goals
    -/


/-- The word of length `m` that alternates between `i` and `i'`, ending with `i'`. -/
def alternatingWord (i i' : B) (m : ℕ) : List B :=
  match m with
  | 0    => []
  | m+1  => (alternatingWord i' i m).concat i'


/-- The word of length `M i i'` that alternates between `i` and `i'`, ending with `i'`. -/
abbrev braidWord (M : CoxeterMatrix B) (i i' : B) : List B := alternatingWord i i' (M i i')


theorem alternatingWord_succ (i i' : B) (m : ℕ) :
    alternatingWord i i' (m + 1) = (alternatingWord i' i m).concat i' := rfl


theorem alternatingWord_succ' (i i' : B) (m : ℕ) :
    alternatingWord i i' (m + 1) = (if Even m then i' else i) :: alternatingWord i i' m := by
  /-
    B : Type u_1
    i i' : B
    m : Nat
    ⊢ Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1)) (List.cons (ite (Eve …
  -/
  induction' m with m ih generalizing i i'
    /-
      case zero
      B : Type u_1
      i i' : B
      ⊢ Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd 0 1)) (List.cons (ite (Eve …
    -/
  · simp [alternatingWord]
    /-
      🎉 no goals
    -/
    /-
      case succ
      B : Type u_1
      m : Nat
      ih : ∀ (i i' : B), Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1)) (Li …
      i i' : B
      ⊢ Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd (HAdd.hAdd m 1) 1)) (List. …
    -/
  · rw [alternatingWord]
    /-
      case succ
      B : Type u_1
      m : Nat
      ih : ∀ (i i' : B), Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1)) (Li …
      i i' : B
      ⊢ Eq ((CoxeterSystem.alternatingWord i' i (HAdd.hAdd m 1)).concat i') (List.co …
    -/
    nth_rw 1 [ih i' i]
    /-
      case succ
      B : Type u_1
      m : Nat
      ih : ∀ (i i' : B), Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1)) (Li …
      i i' : B
      ⊢ Eq ((List.cons (ite (Even m) i i') (CoxeterSystem.alternatingWord i' i m)).c …
    -/
    rw [alternatingWord]
    /-
      case succ
      B : Type u_1
      m : Nat
      ih : ∀ (i i' : B), Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1)) (Li …
      i i' : B
      ⊢ Eq ((List.cons (ite (Even m) i i') (CoxeterSystem.alternatingWord i' i m)).c …
    -/
    simp [Nat.even_add_one, ← Nat.not_even_iff_odd]
    /-
      🎉 no goals
    -/


@[simp]
theorem length_alternatingWord (i i' : B) (m : ℕ) :
    List.length (alternatingWord i i' m) = m := by
  /-
    B : Type u_1
    i i' : B
    m : Nat
    ⊢ Eq (CoxeterSystem.alternatingWord i i' m).length m
  -/
  induction' m with m ih generalizing i i'
    /-
      case zero
      B : Type u_1
      i i' : B
      ⊢ Eq (CoxeterSystem.alternatingWord i i' 0).length 0
    -/
  · dsimp [alternatingWord]
    /-
      🎉 no goals
    -/
    /-
      case succ
      B : Type u_1
      m : Nat
      ih : ∀ (i i' : B), Eq (CoxeterSystem.alternatingWord i i' m).length m
      i i' : B
      ⊢ Eq (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1)).length (HAdd.hAdd m 1)
    -/
  · simpa [alternatingWord] using ih i' i
    /-
      🎉 no goals
    -/


lemma getElem_alternatingWord (i j : B) (p k : ℕ) (hk : k < p) :
                                   /-
                                     B : Type u_1
                                     B' : Type u_2
                                     e : Equiv B B'
                                     W : Type u_3
                                     H : Type u_4
                                     inst✝¹ : Group W
                                     inst✝ : Group H
                                     M : CoxeterMatrix B
                                     cs : CoxeterSystem M W
                                     i j : B
                                     p k : Nat
                                     hk : LT.lt k p
                                     ⊢ LT.lt k (CoxeterSystem.alternatingWord i j p).length
                                   -/
    (alternatingWord i j p)[k]'(by simp; exact hk) =  (if Even (p + k) then i else j) := by
                                         /-
                                           🎉 no goals
                                         -/
  /-
    B : Type u_1
    i j : B
    p k : Nat
    hk : LT.lt k p
    ⊢ Eq (GetElem.getElem (CoxeterSystem.alternatingWord i j p) k ⋯) (ite (Even (H …
  -/
  revert k
  induction p with
  | zero =>
    intro k hk
    simp only [not_lt_zero'] at hk
  | succ n h =>
    intro k hk
    simp_rw [alternatingWord_succ' i j n]
    match k with
    | 0 =>
      by_cases h2 : Even n
      · simp only [h2, ↓reduceIte, getElem_cons_zero, add_zero,
        (by simp [Even.add_one, h2] : ¬Even (n + 1))]
      · simp only [h2, ↓reduceIte, getElem_cons_zero, add_zero,
        Odd.add_one (Nat.not_even_iff_odd.mp h2)]
    | k + 1 =>
      simp only [add_lt_add_iff_right] at hk h
      simp only [getElem_cons_succ, h k hk]
      ring_nf
      have even_add_two (m : ℕ) : Even (2 + m) ↔ Even m := by
        simp only [add_tsub_cancel_right, even_two, (Nat.even_sub (by omega : m ≤ 2 + m)).mp]
      by_cases h_even : Even (n + k)
      · rw [if_pos h_even]
        rw [← even_add_two (n+k), ← Nat.add_assoc 2 n k] at h_even
        rw [if_pos h_even]
      · rw [if_neg h_even]
        rw [← even_add_two (n+k), ← Nat.add_assoc 2 n k] at h_even
        rw [if_neg h_even]


lemma getElem_alternatingWord_swapIndices (i j : B) (p k : ℕ) (h : k + 1 < p) :
                                    /-
                                      B : Type u_1
                                      B' : Type u_2
                                      e : Equiv B B'
                                      W : Type u_3
                                      H : Type u_4
                                      inst✝¹ : Group W
                                      inst✝ : Group H
                                      M : CoxeterMatrix B
                                      cs : CoxeterSystem M W
                                      i j : B
                                      p k : Nat
                                      h : LT.lt (HAdd.hAdd k 1) p
                                      ⊢ LT.lt (HAdd.hAdd k 1) (CoxeterSystem.alternatingWord i j p).length
                                    -/
   (alternatingWord i j p)[k+1]'(by simp; exact h) =
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    B : Type u_1
                                    B' : Type u_2
                                    e : Equiv B B'
                                    W : Type u_3
                                    H : Type u_4
                                    inst✝¹ : Group W
                                    inst✝ : Group H
                                    M : CoxeterMatrix B
                                    cs : CoxeterSystem M W
                                    i j : B
                                    p k : Nat
                                    h : LT.lt (HAdd.hAdd k 1) p
                                    ⊢ LT.lt k (CoxeterSystem.alternatingWord j i p).length
                                  -/
   (alternatingWord j i p)[k]'(by simp [h]; omega) := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    B : Type u_1
    i j : B
    p k : Nat
    h : LT.lt (HAdd.hAdd k 1) p
    ⊢ Eq (GetElem.getElem (CoxeterSystem.alternatingWord i j p) (HAdd.hAdd k 1) ⋯) …
  -/
  rw [getElem_alternatingWord i j p (k+1) (by omega), getElem_alternatingWord j i p k (by omega)]
  /-
    B : Type u_1
    i j : B
    p k : Nat
    h : LT.lt (HAdd.hAdd k 1) p
    ⊢ Eq (ite (Even (HAdd.hAdd p (HAdd.hAdd k 1))) i j) (ite (Even (HAdd.hAdd p k) …
  -/
  by_cases h_even : Even (p + k)
    /-
      case pos
      B : Type u_1
      i j : B
      p k : Nat
      h : LT.lt (HAdd.hAdd k 1) p
      h_even : Even (HAdd.hAdd p k)
      ⊢ Eq (ite (Even (HAdd.hAdd p (HAdd.hAdd k 1))) i j) (ite (Even (HAdd.hAdd p k) …
    -/
  · rw [if_pos h_even, ← add_assoc]
    simp only [ite_eq_right_iff, isEmpty_Prop, Nat.not_even_iff_odd, Even.add_one h_even,
      IsEmpty.forall_iff]
    /-
      case neg
      B : Type u_1
      i j : B
      p k : Nat
      h : LT.lt (HAdd.hAdd k 1) p
      h_even : Not (Even (HAdd.hAdd p k))
      ⊢ Eq (ite (Even (HAdd.hAdd p (HAdd.hAdd k 1))) i j) (ite (Even (HAdd.hAdd p k) …
    -/
  · rw [if_neg h_even, ← add_assoc]
    /-
      case neg
      B : Type u_1
      i j : B
      p k : Nat
      h : LT.lt (HAdd.hAdd k 1) p
      h_even : Not (Even (HAdd.hAdd p k))
      ⊢ Eq (ite (Even (HAdd.hAdd (HAdd.hAdd p k) 1)) i j) i
    -/
    simp [Odd.add_one (Nat.not_even_iff_odd.mp h_even)]
    /-
      🎉 no goals
    -/


lemma listTake_alternatingWord (i j : B) (p k : ℕ) (h : k < 2 * p) :
    List.take k (alternatingWord i j (2 * p)) =
    if Even k then alternatingWord i j k else alternatingWord j i k := by
  induction k with
    | zero =>
      simp only [take_zero, even_zero, ↓reduceIte, alternatingWord]
    | succ k h' =>
      have hk : k < 2 * p := by omega
      apply h' at hk
      by_cases h_even : Even k
      · simp only [h_even, ↓reduceIte] at hk
        simp only [Nat.not_even_iff_odd.mpr (Even.add_one h_even), ↓reduceIte]
        rw [← List.take_concat_get _ _ (by simp[h]; omega), alternatingWord_succ, ← hk]
        apply congr_arg
        rw [getElem_alternatingWord i j (2*p) k (by omega)]
        simp [(by apply Nat.even_add.mpr; simp[h_even]: Even (2 * p + k))]
      · simp only [h_even, ↓reduceIte] at hk
        simp only [(by simp at h_even; exact Odd.add_one h_even : Even (k + 1)), ↓reduceIte]
        rw [← List.take_concat_get _ _ (by simp[h]; omega), alternatingWord_succ, hk]
        apply congr_arg
        rw [getElem_alternatingWord i j (2*p) k (by omega)]
        simp [(by apply Nat.odd_add.mpr; simp[h_even]: Odd (2 * p + k))]


lemma listTake_succ_alternatingWord (i j : B) (p : ℕ) (k : ℕ) (h : k + 1 < 2 * p) :
    List.take (k + 1) (alternatingWord i j (2 * p)) =
    i :: (List.take k (alternatingWord j i (2 * p))) := by
  /-
    B : Type u_1
    i j : B
    p k : Nat
    h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
    ⊢ Eq (List.take (HAdd.hAdd k 1) (CoxeterSystem.alternatingWord i j (HMul.hMul  …
  -/
  rw [listTake_alternatingWord j i p k (by omega), listTake_alternatingWord i j p (k+1) h]

  /-
    B : Type u_1
    i j : B
    p k : Nat
    h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
    ⊢ Eq (ite (Even (HAdd.hAdd k 1)) (CoxeterSystem.alternatingWord i j (HAdd.hAdd …
  -/
  by_cases h_even : Even k
    /-
      case pos
      B : Type u_1
      i j : B
      p k : Nat
      h : LT.lt (HAdd.hAdd k 1) (HMul.hMul 2 p)
      h_even : Even k
      ⊢ Eq (ite (Even (HAdd.hAdd k 1)) (CoxeterSystem.alternatingWord i j (HAdd.hAdd …
    -/
  · simp [h_even, Nat.not_even_iff_odd.mpr (Even.add_one h_even), alternatingWord_succ', h_even]
    /-
      🎉 no goals
    -/
  · simp [h_even, (by simp at h_even; exact Odd.add_one h_even: Even (k + 1)),
    alternatingWord_succ', h_even]


theorem prod_alternatingWord_eq_mul_pow (i i' : B) (m : ℕ) :
    π (alternatingWord i i' m) = (if Even m then 1 else s i') * (s i * s i') ^ (m / 2) := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    ⊢ Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite (Eve …
  -/
  induction' m with m ih
    /-
      case zero
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      ⊢ Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' 0)) (HMul.hMul (ite (Eve …
    -/
  · simp [alternatingWord]
    /-
      🎉 no goals
    -/
    /-
      case succ
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
      ⊢ Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' (HAdd.hAdd m 1))) (HMul. …
    -/
  · rw [alternatingWord_succ', wordProd_cons, ih]
    /-
      case succ
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
      ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
    -/
    by_cases hm : Even m
      /-
        case pos
        B : Type u_1
        W : Type u_3
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i i' : B
        m : Nat
        ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
        hm : Even m
        ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
      -/
    · have h₁ : ¬ Even (m + 1) := by simp [hm, parity_simps]
      /-
        case pos
        B : Type u_1
        W : Type u_3
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i i' : B
        m : Nat
        ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
        hm : Even m
        h₁ : Not (Even (HAdd.hAdd m 1))
        ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
      -/
      have h₂ : (m + 1) / 2 = m / 2 := Nat.succ_div_of_not_dvd <| by rwa [← even_iff_two_dvd]
      /-
        case pos
        B : Type u_1
        W : Type u_3
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i i' : B
        m : Nat
        ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
        hm : Even m
        h₁ : Not (Even (HAdd.hAdd m 1))
        h₂ : Eq (HDiv.hDiv (HAdd.hAdd m 1) 2) (HDiv.hDiv m 2)
        ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
      -/
      simp [hm, h₁, h₂]
      /-
        🎉 no goals
      -/
      /-
        case neg
        B : Type u_1
        W : Type u_3
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i i' : B
        m : Nat
        ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
        hm : Not (Even m)
        ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
      -/
    · have h₁ : Even (m + 1) := by simp [hm, parity_simps]
      /-
        case neg
        B : Type u_1
        W : Type u_3
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i i' : B
        m : Nat
        ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
        hm : Not (Even m)
        h₁ : Even (HAdd.hAdd m 1)
        ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
      -/
      have h₂ : (m + 1) / 2 = m / 2 + 1 := Nat.succ_div_of_dvd h₁.two_dvd
      /-
        case neg
        B : Type u_1
        W : Type u_3
        inst✝ : Group W
        M : CoxeterMatrix B
        cs : CoxeterSystem M W
        i i' : B
        m : Nat
        ih : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (HMul.hMul (ite ( …
        hm : Not (Even m)
        h₁ : Even (HAdd.hAdd m 1)
        h₂ : Eq (HDiv.hDiv (HAdd.hAdd m 1) 2) (HAdd.hAdd (HDiv.hDiv m 2) 1)
        ⊢ Eq (HMul.hMul (cs.simple (ite (Even m) i' i)) (HMul.hMul (ite (Even m) 1 (cs …
      -/
      simp [hm, h₁, h₂, ← pow_succ', ← mul_assoc]
      /-
        🎉 no goals
      -/


theorem prod_alternatingWord_eq_prod_alternatingWord_sub (i i' : B) (m : ℕ) (hm : m ≤ M i i' * 2) :
    π (alternatingWord i i' m) = π (alternatingWord i' i (M i i' * 2 - m)) := by
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    hm : LE.le m (HMul.hMul (M.M i i') 2)
    ⊢ Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' m)) (cs.wordProd (Coxete …
  -/
  simp_rw [prod_alternatingWord_eq_mul_pow, ← Int.even_coe_nat]

  /- Rewrite everything in terms of an integer m' which is equal to m.
  The resulting equation holds for all integers m'. -/
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    hm : LE.le m (HMul.hMul (M.M i i') 2)
    ⊢ Eq (HMul.hMul (ite (Even ↑m) 1 (cs.simple i')) (HPow.hPow (HMul.hMul (cs.sim …
  -/
  simp_rw [← zpow_natCast, Int.ofNat_ediv, Int.ofNat_sub hm]
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    hm : LE.le m (HMul.hMul (M.M i i') 2)
    ⊢ Eq (HMul.hMul (ite (Even ↑m) 1 (cs.simple i')) (HPow.hPow (HMul.hMul (cs.sim …
  -/
  generalize (m : ℤ) = m'
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    hm : LE.le m (HMul.hMul (M.M i i') 2)
    m' : Int
    ⊢ Eq (HMul.hMul (ite (Even m') 1 (cs.simple i')) (HPow.hPow (HMul.hMul (cs.sim …
  -/
  clear hm
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    m' : Int
    ⊢ Eq (HMul.hMul (ite (Even m') 1 (cs.simple i')) (HPow.hPow (HMul.hMul (cs.sim …
  -/
  push_cast

  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    m : Nat
    m' : Int
    ⊢ Eq (HMul.hMul (ite (Even m') 1 (cs.simple i')) (HPow.hPow (HMul.hMul (cs.sim …
  -/
  rcases Int.even_or_odd' m' with ⟨k, rfl | rfl⟩
    /-
      case intro.inl
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      ⊢ Eq (HMul.hMul (ite (Even (HMul.hMul 2 k)) 1 (cs.simple i')) (HPow.hPow (HMul …
    -/
  · rw [if_pos (by use k; ring), if_pos (by use -k + (M i i'); ring), mul_comm 2 k, ← sub_mul]
    /-
      case intro.inl
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      ⊢ Eq (HMul.hMul 1 (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple i')) (HDiv.hD …
    -/
    repeat rw [Int.mul_ediv_cancel _ (by norm_num)]
    /-
      case intro.inl
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      ⊢ Eq (HMul.hMul 1 (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple i')) k)) (HMu …
    -/
    rw [zpow_sub, zpow_natCast, simple_mul_simple_pow' cs i i', ← inv_zpow]
    /-
      case intro.inl
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      ⊢ Eq (HMul.hMul 1 (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple i')) k)) (HMu …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      ⊢ Eq (HMul.hMul (ite (Even (HAdd.hAdd (HMul.hMul 2 k) 1)) 1 (cs.simple i')) (H …
    -/
  · have : ¬Even (2 * k + 1) := Int.not_even_iff_odd.2 ⟨k, rfl⟩
    /-
      case intro.inr
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      this : Not (Even (HAdd.hAdd (HMul.hMul 2 k) 1))
      ⊢ Eq (HMul.hMul (ite (Even (HAdd.hAdd (HMul.hMul 2 k) 1)) 1 (cs.simple i')) (H …
    -/
    rw [if_neg this]
    have : ¬Even (↑(M i i') * 2 - (2 * k + 1)) :=
      Int.not_even_iff_odd.2 ⟨↑(M i i') - k - 1, by ring⟩
    /-
      case intro.inr
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      this✝ : Not (Even (HAdd.hAdd (HMul.hMul 2 k) 1))
      this : Not (Even (HSub.hSub (HMul.hMul (↑(M.M i i')) 2) (HAdd.hAdd (HMul.hMul  …
      ⊢ Eq (HMul.hMul (cs.simple i') (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple  …
    -/
    rw [if_neg this]

    rw [(by ring : ↑(M i i') * 2 - (2 * k + 1) = -1 + (-k + ↑(M i i')) * 2),
      (by ring : 2 * k + 1 = 1 + k * 2)]
    /-
      case intro.inr
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      this✝ : Not (Even (HAdd.hAdd (HMul.hMul 2 k) 1))
      this : Not (Even (HSub.hSub (HMul.hMul (↑(M.M i i')) 2) (HAdd.hAdd (HMul.hMul  …
      ⊢ Eq (HMul.hMul (cs.simple i') (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple  …
    -/
    repeat rw [Int.add_mul_ediv_right _ _ (by norm_num)]
    /-
      case intro.inr
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      this✝ : Not (Even (HAdd.hAdd (HMul.hMul 2 k) 1))
      this : Not (Even (HSub.hSub (HMul.hMul (↑(M.M i i')) 2) (HAdd.hAdd (HMul.hMul  …
      ⊢ Eq (HMul.hMul (cs.simple i') (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple  …
    -/
    norm_num

    rw [zpow_add, zpow_add, zpow_natCast, simple_mul_simple_pow', zpow_neg, ← inv_zpow, zpow_neg,
      ← inv_zpow]
    /-
      case intro.inr
      B : Type u_1
      W : Type u_3
      inst✝ : Group W
      M : CoxeterMatrix B
      cs : CoxeterSystem M W
      i i' : B
      m : Nat
      k : Int
      this✝ : Not (Even (HAdd.hAdd (HMul.hMul 2 k) 1))
      this : Not (Even (HSub.hSub (HMul.hMul (↑(M.M i i')) 2) (HAdd.hAdd (HMul.hMul  …
      ⊢ Eq (HMul.hMul (cs.simple i') (HPow.hPow (HMul.hMul (cs.simple i) (cs.simple  …
    -/
    simp [← mul_assoc]
    /-
      🎉 no goals
    -/


/-- The two words of length `M i i'` that alternate between `i` and `i'` have the same product.
This is known as the "braid relation" or "Artin-Tits relation". -/
theorem wordProd_braidWord_eq (i i' : B) :
    π (braidWord M i i') = π (braidWord M i' i) := by
  have := cs.prod_alternatingWord_eq_prod_alternatingWord_sub i i' (M i i')
    (Nat.le_mul_of_pos_right _ (by norm_num))
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    this : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' (M.M i i'))) (cs.wo …
    ⊢ Eq (cs.wordProd (CoxeterSystem.braidWord M i i')) (cs.wordProd (CoxeterSyste …
  -/
  rw [tsub_eq_of_eq_add (mul_two (M i i'))] at this
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    this : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' (M.M i i'))) (cs.wo …
    ⊢ Eq (cs.wordProd (CoxeterSystem.braidWord M i i')) (cs.wordProd (CoxeterSyste …
  -/
  nth_rw 2 [M.symmetric i i'] at this
  /-
    B : Type u_1
    W : Type u_3
    inst✝ : Group W
    M : CoxeterMatrix B
    cs : CoxeterSystem M W
    i i' : B
    this : Eq (cs.wordProd (CoxeterSystem.alternatingWord i i' (M.M i i'))) (cs.wo …
    ⊢ Eq (cs.wordProd (CoxeterSystem.braidWord M i i')) (cs.wordProd (CoxeterSyste …
  -/
  exact this
  /-
    🎉 no goals
  -/


