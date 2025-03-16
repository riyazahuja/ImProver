/-- The average of all elements of the group `G`, considered as an element of `MonoidAlgebra k G`.
-/
noncomputable def average : MonoidAlgebra k G :=
  ⅟ (Fintype.card G : k) • ∑ g : G, of k G g


/-- `average k G` is invariant under left multiplication by elements of `G`.
-/
@[simp]
theorem mul_average_left (g : G) : ↑(Finsupp.single g 1) * average k G = average k G := by
  simp only [mul_one, Finset.mul_sum, Algebra.mul_smul_comm, average, MonoidAlgebra.of_apply,
    Finset.sum_congr, MonoidAlgebra.single_mul_single]
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    g : G
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun x  …
  -/
  set f : G → MonoidAlgebra k G := fun x => Finsupp.single x 1
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    g : G
    f : G → MonoidAlgebra k G := fun x => Finsupp.single x 1
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun x  …
  -/
  show ⅟ (Fintype.card G : k) • ∑ x : G, f (g * x) = ⅟ (Fintype.card G : k) • ∑ x : G, f x
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    g : G
    f : G → MonoidAlgebra k G := fun x => Finsupp.single x 1
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun x  …
  -/
  rw [Function.Bijective.sum_comp (Group.mulLeft_bijective g) _]
  /-
    🎉 no goals
  -/


/-- `average k G` is invariant under right multiplication by elements of `G`.
-/
@[simp]
theorem mul_average_right (g : G) : average k G * ↑(Finsupp.single g 1) = average k G := by
  simp only [mul_one, Finset.sum_mul, Algebra.smul_mul_assoc, average, MonoidAlgebra.of_apply,
    Finset.sum_congr, MonoidAlgebra.single_mul_single]
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    g : G
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun x  …
  -/
  set f : G → MonoidAlgebra k G := fun x => Finsupp.single x 1
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    g : G
    f : G → MonoidAlgebra k G := fun x => Finsupp.single x 1
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun x  …
  -/
  show ⅟ (Fintype.card G : k) • ∑ x : G, f (x * g) = ⅟ (Fintype.card G : k) • ∑ x : G, f x
  /-
    k : Type u_1
    G : Type u_2
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    g : G
    f : G → MonoidAlgebra k G := fun x => Finsupp.single x 1
    ⊢ Eq (HSMul.hSMul (Invertible.invOf ↑(Fintype.card G)) (Finset.univ.sum fun x  …
  -/
  rw [Function.Bijective.sum_comp (Group.mulRight_bijective g) _]
  /-
    🎉 no goals
  -/


/-- The subspace of invariants, consisting of the vectors fixed by all elements of `G`.
-/
def invariants : Submodule k V where
  carrier := setOf fun v => ∀ g : G, ρ g v = v
                    /-
                      k : Type u_1
                      G : Type u_2
                      V : Type u_3
                      inst✝³ : CommSemiring k
                      inst✝² : Group G
                      inst✝¹ : AddCommMonoid V
                      inst✝ : Module k V
                      ρ : Representation k G V
                      g : G
                      ⊢ Eq ((ρ g) 0) 0
                    -/
                         /-
                           k : Type u_1
                           G : Type u_2
                           V : Type u_3
                           inst✝³ : CommSemiring k
                           inst✝² : Group G
                           inst✝¹ : AddCommMonoid V
                           inst✝ : Module k V
                           ρ : Representation k G V
                           a✝ b✝ : V
                           hv : Membership.mem (setOf fun v => ∀ (g : G), Eq ((ρ g) v) v) a✝
                           hw : Membership.mem (setOf fun v => ∀ (g : G), Eq ((ρ g) v) v) b✝
                           g : G
                           ⊢ Eq ((ρ g) (HAdd.hAdd a✝ b✝)) (HAdd.hAdd a✝ b✝)
                         -/
  zero_mem' g := by simp only [map_zero]
                         /-
                           🎉 no goals
                         -/
                    /-
                      🎉 no goals
                    -/
  add_mem' hv hw g := by simp only [hv g, hw g, map_add]
                           /-
                             k : Type u_1
                             G : Type u_2
                             V : Type u_3
                             inst✝³ : CommSemiring k
                             inst✝² : Group G
                             inst✝¹ : AddCommMonoid V
                             inst✝ : Module k V
                             ρ : Representation k G V
                             r : k
                             v : V
                             hv : Membership.mem { carrier := setOf fun v => ∀ (g : G), Eq ((ρ g) v) v, add …
                             g : G
                             ⊢ Eq ((ρ g) (HSMul.hSMul r v)) (HSMul.hSMul r v)
                           -/
  smul_mem' r v hv g := by simp only [hv g, LinearMap.map_smulₛₗ, RingHom.id_apply]
                           /-
                             🎉 no goals
                           -/


@[simp]
                                                                             /-
                                                                               k : Type u_1
                                                                               G : Type u_2
                                                                               V : Type u_3
                                                                               inst✝³ : CommSemiring k
                                                                               inst✝² : Group G
                                                                               inst✝¹ : AddCommMonoid V
                                                                               inst✝ : Module k V
                                                                               ρ : Representation k G V
                                                                               v : V
                                                                               ⊢ Iff (Membership.mem ρ.invariants v) (∀ (g : G), Eq ((ρ g) v) v)
                                                                             -/
theorem mem_invariants (v : V) : v ∈ invariants ρ ↔ ∀ g : G, ρ g v = v := by rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem invariants_eq_inter : (invariants ρ).carrier = ⋂ g : G, Function.fixedPoints (ρ g) := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝³ : CommSemiring k
    inst✝² : Group G
    inst✝¹ : AddCommMonoid V
    inst✝ : Module k V
    ρ : Representation k G V
    ⊢ Eq ρ.invariants.carrier (Set.iInter fun g => Function.fixedPoints ⇑(ρ g))
  -/
  ext; simp [Function.IsFixedPt]
       /-
         🎉 no goals
       -/


theorem invariants_eq_top [ρ.IsTrivial] :
    invariants ρ = ⊤ :=
eq_top_iff.2 (fun x _ g => ρ.apply_eq_self g x)


/-- The action of `average k G` gives a projection map onto the subspace of invariants.
-/
@[simp]
noncomputable def averageMap : V →ₗ[k] V :=
  asAlgebraHom ρ (average k G)


/-- The `averageMap` sends elements of `V` to the subspace of invariants.
-/
theorem averageMap_invariant (v : V) : averageMap ρ v ∈ invariants ρ := fun g => by
  rw [averageMap, ← asAlgebraHom_single_one, ← LinearMap.mul_apply, ← map_mul (asAlgebraHom ρ),
    mul_average_left]


/-- The `averageMap` acts as the identity on the subspace of invariants.
-/
theorem averageMap_id (v : V) (hv : v ∈ invariants ρ) : averageMap ρ v = v := by
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Group G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    ρ : Representation k G V
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    v : V
    hv : Membership.mem ρ.invariants v
    ⊢ Eq (ρ.averageMap v) v
  -/
  rw [mem_invariants] at hv
  /-
    k : Type u_1
    G : Type u_2
    V : Type u_3
    inst✝⁵ : CommSemiring k
    inst✝⁴ : Group G
    inst✝³ : AddCommMonoid V
    inst✝² : Module k V
    ρ : Representation k G V
    inst✝¹ : Fintype G
    inst✝ : Invertible ↑(Fintype.card G)
    v : V
    hv : ∀ (g : G), Eq ((ρ g) v) v
    ⊢ Eq (ρ.averageMap v) v
  -/
  simp [average, map_sum, hv, Finset.card_univ, ← Nat.cast_smul_eq_nsmul k _ v, smul_smul]
  /-
    🎉 no goals
  -/


theorem isProj_averageMap : LinearMap.IsProj ρ.invariants ρ.averageMap :=
  ⟨ρ.averageMap_invariant, ρ.averageMap_id⟩


theorem mem_invariants_iff_comm {X Y : Rep k G} (f : X.V →ₗ[k] Y.V) (g : G) :
    (linHom X.ρ Y.ρ) g f = f ↔ f.comp (X.ρ g) = (Y.ρ g).comp f := by
  /-
    k : Type u
    inst✝ : CommRing k
    G : Grp
    X Y : Rep k ↑G
    f : LinearMap (RingHom.id k) ↑X.V ↑Y.V
    g : ↑G
    ⊢ Iff (Eq (((X.ρ.linHom Y.ρ) g) f) f) (Eq (f.comp (X.ρ g)) ((Y.ρ g).comp f))
  -/
  dsimp
  rw [← LinearMap.comp_assoc, ← ModuleCat.hom_ofHom (Y.ρ g), ← ModuleCat.hom_ofHom f,
      ← ModuleCat.hom_comp, ← ModuleCat.hom_ofHom (X.ρ g⁻¹), ← ModuleCat.hom_comp,
      Rep.ofHom_ρ, ← ρAut_apply_inv X g, Rep.ofHom_ρ, ← ρAut_apply_hom Y g, ← ModuleCat.hom_ext_iff,
      Iso.inv_comp_eq, ρAut_apply_hom, ← ModuleCat.hom_ofHom (X.ρ g),
      ← ModuleCat.hom_comp, ← ModuleCat.hom_ext_iff]
  /-
    k : Type u
    inst✝ : CommRing k
    G : Grp
    X Y : Rep k ↑G
    f : LinearMap (RingHom.id k) ↑X.V ↑Y.V
    g : ↑G
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom f) (Y.ρ g)) (Ca …
  -/
  exact comm
  /-
    🎉 no goals
  -/


/-- The invariants of the representation `linHom X.ρ Y.ρ` correspond to the representation
homomorphisms from `X` to `Y`. -/
@[simps]
def invariantsEquivRepHom (X Y : Rep k G) : (linHom X.ρ Y.ρ).invariants ≃ₗ[k] X ⟶ Y where
  toFun f := ⟨ModuleCat.ofHom f.val, fun g =>
    ModuleCat.hom_ext ((mem_invariants_iff_comm _ g).1 (f.property g))⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  invFun f := ⟨f.hom.hom, fun g =>
    (mem_invariants_iff_comm _ g).2 (ModuleCat.hom_ext_iff.mp (f.comm g))⟩
                   /-
                     k : Type u
                     inst✝ : CommRing k
                     G : Grp
                     X Y : Rep k ↑G
                     x✝ : Subtype fun x => Membership.mem (X.ρ.linHom Y.ρ).invariants x
                     ⊢ Eq ((fun f => ⟨f.hom.hom, ⋯⟩) ({ toFun := fun f => { hom := ModuleCat.ofHom  …
                   -/
  left_inv _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      k : Type u
                      inst✝ : CommRing k
                      G : Grp
                      X Y : Rep k ↑G
                      x✝ : Quiver.Hom X Y
                      ⊢ Eq ({ toFun := fun f => { hom := ModuleCat.ofHom ↑f, comm := ⋯ }, map_add' : …
                    -/
  right_inv _ := by ext; rfl
                         /-
                           🎉 no goals
                         -/


/-- The invariants of the representation `linHom X.ρ Y.ρ` correspond to the representation
homomorphisms from `X` to `Y`. -/
def invariantsEquivFDRepHom (X Y : FDRep k G) : (linHom X.ρ Y.ρ).invariants ≃ₗ[k] X ⟶ Y := by
  /-
    k : Type u
    inst✝ : Field k
    G : Grp
    X Y : FDRep k ↑G
    ⊢ LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Representation. …
  -/
  rw [← FDRep.forget₂_ρ, ← FDRep.forget₂_ρ]
  -- Porting note: The original version used `linHom.invariantsEquivRepHom _ _ ≪≫ₗ`
  exact linHom.invariantsEquivRepHom
    ((forget₂ (FDRep k G) (Rep k G)).obj X) ((forget₂ (FDRep k G) (Rep k G)).obj Y) ≪≫ₗ
    FDRep.forget₂HomLinearEquiv X Y


