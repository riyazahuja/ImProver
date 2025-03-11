/-- The (absolute) polar of `s : Set E` is given by the set of all `y : F` such that `‖B x y‖ ≤ 1`
for all `x ∈ s`. -/
def polar (s : Set E) : Set F :=
  { y : F | ∀ x ∈ s, ‖B x y‖ ≤ 1 }


theorem polar_mem_iff (s : Set E) (y : F) : y ∈ B.polar s ↔ ∀ x ∈ s, ‖B x y‖ ≤ 1 :=
  Iff.rfl


theorem polar_mem (s : Set E) (y : F) (hy : y ∈ B.polar s) : ∀ x ∈ s, ‖B x y‖ ≤ 1 :=
  hy


@[simp]
theorem zero_mem_polar (s : Set E) : (0 : F) ∈ B.polar s := fun _ _ => by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x✝¹ : E
    x✝ : Membership.mem s x✝¹
    ⊢ LE.le (Norm.norm ((B x✝¹) 0)) 1
  -/
  simp only [map_zero, norm_zero, zero_le_one]
  /-
    🎉 no goals
  -/


theorem polar_nonempty (s : Set E) : Set.Nonempty (B.polar s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    ⊢ (B.polar s).Nonempty
  -/
  use 0
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    ⊢ Membership.mem (B.polar s) 0
  -/
  exact zero_mem_polar B s
  /-
    🎉 no goals
  -/


theorem polar_eq_iInter {s : Set E} : B.polar s = ⋂ x ∈ s, { y : F | ‖B x y‖ ≤ 1 } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    ⊢ Eq (B.polar s) (Set.iInter fun x => Set.iInter fun h => setOf fun y => LE.le …
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x✝ : F
    ⊢ Iff (Membership.mem (B.polar s) x✝) (Membership.mem (Set.iInter fun x => Set …
  -/
  simp only [polar_mem_iff, Set.mem_iInter, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


/-- The map `B.polar : Set E → Set F` forms an order-reversing Galois connection with
`B.flip.polar : Set F → Set E`. We use `OrderDual.toDual` and `OrderDual.ofDual` to express
that `polar` is order-reversing. -/
theorem polar_gc :
    GaloisConnection (OrderDual.toDual ∘ B.polar) (B.flip.polar ∘ OrderDual.ofDual) := fun _ _ =>
  ⟨fun h _ hx _ hy => h hy _ hx, fun h _ hx _ hy => h hy _ hx⟩


@[simp]
theorem polar_iUnion {ι} {s : ι → Set E} : B.polar (⋃ i, s i) = ⋂ i, B.polar (s i) :=
  B.polar_gc.l_iSup


@[simp]
theorem polar_union {s t : Set E} : B.polar (s ∪ t) = B.polar s ∩ B.polar t :=
  B.polar_gc.l_sup


theorem polar_antitone : Antitone (B.polar : Set E → Set F) :=
  B.polar_gc.monotone_l


@[simp]
theorem polar_empty : B.polar ∅ = Set.univ :=
  B.polar_gc.l_bot


@[simp]
theorem polar_singleton {a : E} : B.polar {a} = { y | ‖B a y‖ ≤ 1 } := le_antisymm
  (fun _ hy => hy _ rfl)
                                                        /-
                                                          𝕜 : Type u_1
                                                          E : Type u_2
                                                          F : Type u_3
                                                          inst✝⁴ : NormedCommRing 𝕜
                                                          inst✝³ : AddCommMonoid E
                                                          inst✝² : AddCommMonoid F
                                                          inst✝¹ : Module 𝕜 E
                                                          inst✝ : Module 𝕜 F
                                                          B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
                                                          a : E
                                                          y : F
                                                          hy : Membership.mem (setOf fun y => LE.le (Norm.norm ((B a) y)) 1) y
                                                          x✝ : E
                                                          hb : Membership.mem (Singleton.singleton a) x✝
                                                          ⊢ LE.le (Norm.norm ((B x✝) y)) 1
                                                        -/
  (fun y hy => (polar_mem_iff _ _ _).mp (fun _ hb => by rw [Set.mem_singleton_iff.mp hb]; exact hy))
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem mem_polar_singleton {x : E} (y : F) : y ∈ B.polar {x} ↔ ‖B x y‖ ≤ 1 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    x : E
    y : F
    ⊢ Iff (Membership.mem (B.polar (Singleton.singleton x)) y) (LE.le (Norm.norm ( …
  -/
  simp only [polar_singleton, Set.mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem polar_zero : B.polar ({0} : Set E) = Set.univ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    ⊢ Eq (B.polar (Singleton.singleton 0)) Set.univ
  -/
  simp only [polar_singleton, map_zero, zero_apply, norm_zero, zero_le_one, Set.setOf_true]
  /-
    🎉 no goals
  -/


theorem subset_bipolar (s : Set E) : s ⊆ B.flip.polar (B.polar s) := fun x hx y hy => by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x : E
    hx : Membership.mem s x
    y : F
    hy : Membership.mem (B.polar s) y
    ⊢ LE.le (Norm.norm ((B.flip y) x)) 1
  -/
  rw [B.flip_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x : E
    hx : Membership.mem s x
    y : F
    hy : Membership.mem (B.polar s) y
    ⊢ LE.le (Norm.norm ((B x) y)) 1
  -/
  exact hy x hx
  /-
    🎉 no goals
  -/


@[simp]
theorem tripolar_eq_polar (s : Set E) : B.polar (B.flip.polar (B.polar s)) = B.polar s :=
  (B.polar_antitone (B.subset_bipolar s)).antisymm (subset_bipolar B.flip (B.polar s))


/-- The polar set is closed in the weak topology induced by `B.flip`. -/
theorem polar_weak_closed (s : Set E) : IsClosed[WeakBilin.instTopologicalSpace B.flip]
    (B.polar s) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    ⊢ IsClosed (B.polar s)
  -/
  rw [polar_eq_iInter]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    ⊢ IsClosed (Set.iInter fun x => Set.iInter fun h => setOf fun y => LE.le (Norm …
  -/
  refine isClosed_iInter fun x => isClosed_iInter fun _ => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x : E
    x✝ : Membership.mem s x
    ⊢ IsClosed (setOf fun y => LE.le (Norm.norm ((B x) y)) 1)
  -/
  exact isClosed_le (WeakBilin.eval_continuous B.flip x).norm continuous_const
  /-
    🎉 no goals
  -/


theorem sInter_polar_finite_subset_eq_polar (s : Set E) :
    ⋂₀ (B.polar '' { F | F.Finite ∧ F ⊆ s }) = B.polar s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    ⊢ Eq (Set.image B.polar (setOf fun F => And F.Finite (HasSubset.Subset F s))). …
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x : F
    ⊢ Iff (Membership.mem (Set.image B.polar (setOf fun F => And F.Finite (HasSubs …
  -/
  simp only [Set.sInter_image, Set.mem_setOf_eq, Set.mem_iInter, and_imp]
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x : F
    ⊢ Iff (∀ (i : Set E), i.Finite → HasSubset.Subset i s → Membership.mem (B.pola …
  -/
  refine ⟨fun hx a ha ↦ ?_, fun hx F _ hF₂ => polar_antitone _ hF₂ hx⟩
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedCommRing 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    s : Set E
    x : F
    hx : ∀ (i : Set E), i.Finite → HasSubset.Subset i s → Membership.mem (B.polar  …
    a : E
    ha : Membership.mem s a
    ⊢ LE.le (Norm.norm ((B a) x)) 1
  -/
  simpa [mem_polar_singleton] using hx _ (Set.finite_singleton a) (Set.singleton_subset_iff.mpr ha)
  /-
    🎉 no goals
  -/


theorem polar_univ (h : SeparatingRight B) : B.polar Set.univ = {(0 : F)} := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    h : B.SeparatingRight
    ⊢ Eq (B.polar Set.univ) (Singleton.singleton 0)
  -/
  rw [Set.eq_singleton_iff_unique_mem]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    h : B.SeparatingRight
    ⊢ And (Membership.mem (B.polar Set.univ) 0) (∀ (x : F), Membership.mem (B.pola …
  -/
  refine ⟨by simp only [zero_mem_polar], fun y hy => h _ fun x => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    h : B.SeparatingRight
    y : F
    hy : Membership.mem (B.polar Set.univ) y
    x : E
    ⊢ Eq ((B x) y) 0
  -/
  refine norm_le_zero_iff.mp (le_of_forall_le_of_dense fun ε hε => ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 E
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    h : B.SeparatingRight
    y : F
    hy : Membership.mem (B.polar Set.univ) y
    x : E
    ε : Real
    hε : LT.lt 0 ε
    ⊢ LE.le (Norm.norm ((B x) y)) ε
  -/
  rcases NormedField.exists_norm_lt 𝕜 hε with ⟨c, hc, hcε⟩
  calc
    ‖B x y‖ = ‖c‖ * ‖B (c⁻¹ • x) y‖ := by
      rw [B.map_smul, LinearMap.smul_apply, Algebra.id.smul_eq_mul, norm_mul, norm_inv,
        mul_inv_cancel_left₀ hc.ne']
    _ ≤ ε * 1 := by gcongr; exact hy _ trivial
    _ = ε := mul_one _


theorem polar_subMulAction {S : Type*} [SetLike S E] [SMulMemClass S 𝕜 E] (m : S) :
    B.polar m = { y | ∀ x ∈ m, B x y = 0 } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommMonoid E
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module 𝕜 E
    inst✝² : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    S : Type u_4
    inst✝¹ : SetLike S E
    inst✝ : SMulMemClass S 𝕜 E
    m : S
    ⊢ Eq (B.polar ↑m) (setOf fun y => ∀ (x : E), Membership.mem m x → Eq ((B x) y) …
  -/
  ext y
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : AddCommMonoid E
    inst✝⁴ : AddCommMonoid F
    inst✝³ : Module 𝕜 E
    inst✝² : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    S : Type u_4
    inst✝¹ : SetLike S E
    inst✝ : SMulMemClass S 𝕜 E
    m : S
    y : F
    ⊢ Iff (Membership.mem (B.polar ↑m) y) (Membership.mem (setOf fun y => ∀ (x : E …
  -/
  constructor
    /-
      case h.mp
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      ⊢ Membership.mem (B.polar ↑m) y → Membership.mem (setOf fun y => ∀ (x : E), Me …
    -/
  · intro hy x hx
    /-
      case h.mp
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      hy : Membership.mem (B.polar ↑m) y
      x : E
      hx : Membership.mem m x
      ⊢ Eq ((B x) y) 0
    -/
    obtain ⟨r, hr⟩ := NormedField.exists_lt_norm 𝕜 ‖B x y‖⁻¹
    /-
      case h.mp.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      hy : Membership.mem (B.polar ↑m) y
      x : E
      hx : Membership.mem m x
      r : 𝕜
      hr : LT.lt (Inv.inv (Norm.norm ((B x) y))) (Norm.norm r)
      ⊢ Eq ((B x) y) 0
    -/
    contrapose! hr
    /-
      case h.mp.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      hy : Membership.mem (B.polar ↑m) y
      x : E
      hx : Membership.mem m x
      r : 𝕜
      hr : Ne ((B x) y) 0
      ⊢ LE.le (Norm.norm r) (Inv.inv (Norm.norm ((B x) y)))
    -/
    rw [← one_div, le_div_iff₀ (norm_pos_iff.2 hr)]
    /-
      case h.mp.intro
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      hy : Membership.mem (B.polar ↑m) y
      x : E
      hx : Membership.mem m x
      r : 𝕜
      hr : Ne ((B x) y) 0
      ⊢ LE.le (HMul.hMul (Norm.norm r) (Norm.norm ((B x) y))) 1
    -/
    simpa using hy _ (SMulMemClass.smul_mem r hx)
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      ⊢ Membership.mem (setOf fun y => ∀ (x : E), Membership.mem m x → Eq ((B x) y)  …
    -/
  · intro h x hx
    /-
      case h.mpr
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : AddCommMonoid E
      inst✝⁴ : AddCommMonoid F
      inst✝³ : Module 𝕜 E
      inst✝² : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      S : Type u_4
      inst✝¹ : SetLike S E
      inst✝ : SMulMemClass S 𝕜 E
      m : S
      y : F
      h : Membership.mem (setOf fun y => ∀ (x : E), Membership.mem m x → Eq ((B x) y …
      x : E
      hx : Membership.mem (↑m) x
      ⊢ LE.le (Norm.norm ((B x) y)) 1
    -/
    simp [h x hx]
    /-
      🎉 no goals
    -/


/-- The polar of a set closed under scalar multiplication as a submodule -/
def polarSubmodule {S : Type*} [SetLike S E] [SMulMemClass S 𝕜 E] (m : S) : Submodule 𝕜 F :=
                                                         /-
                                                           𝕜 : Type u_1
                                                           E : Type u_2
                                                           F : Type u_3
                                                           inst✝⁶ : NontriviallyNormedField 𝕜
                                                           inst✝⁵ : AddCommMonoid E
                                                           inst✝⁴ : AddCommMonoid F
                                                           inst✝³ : Module 𝕜 E
                                                           inst✝² : Module 𝕜 F
                                                           B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
                                                           S : Type u_4
                                                           inst✝¹ : SetLike S E
                                                           inst✝ : SMulMemClass S 𝕜 E
                                                           m : S
                                                           ⊢ Eq (B.polar ↑m) ↑(iInf fun x => iInf fun h => LinearMap.ker (B x))
                                                         -/
  .copy (⨅ x ∈ m, LinearMap.ker (B x)) (B.polar m) <| by ext; simp [polar_subMulAction]
                                                              /-
                                                                🎉 no goals
                                                              -/


