theorem Set.star_mem_center (ha : a ∈ Set.center R) : star a ∈ Set.center R where
  comm := by simpa only [star_mul, star_star] using fun g =>
    congr_arg star ((mem_center_iff.1 ha).comm <| star g).symm
  left_assoc b c := calc
                                                                      /-
                                                                        R : Type u_1
                                                                        inst✝¹ : Mul R
                                                                        inst✝ : StarMul R
                                                                        a : R
                                                                        ha : Membership.mem (Set.center R) a
                                                                        b c : R
                                                                        ⊢ Eq (HMul.hMul (Star.star a) (HMul.hMul b c)) (HMul.hMul (Star.star a) (HMul. …
                                                                      -/
    star a * (b * c) = star a * (star (star b) * star (star c)) := by rw [star_star, star_star]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                              /-
                                                R : Type u_1
                                                inst✝¹ : Mul R
                                                inst✝ : StarMul R
                                                a : R
                                                ha : Membership.mem (Set.center R) a
                                                b c : R
                                                ⊢ Eq (HMul.hMul (Star.star a) (HMul.hMul (Star.star (Star.star b)) (Star.star  …
                                              -/
    _ = star a * star (star c * star b) := by rw [star_mul]
                                              /-
                                                🎉 no goals
                                              -/
                                           /-
                                             R : Type u_1
                                             inst✝¹ : Mul R
                                             inst✝ : StarMul R
                                             a : R
                                             ha : Membership.mem (Set.center R) a
                                             b c : R
                                             ⊢ Eq (HMul.hMul (Star.star a) (Star.star (HMul.hMul (Star.star c) (Star.star b …
                                           -/
    _ = star ((star c * star b) * a) := by rw [← star_mul]
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             R : Type u_1
                                             inst✝¹ : Mul R
                                             inst✝ : StarMul R
                                             a : R
                                             ha : Membership.mem (Set.center R) a
                                             b c : R
                                             ⊢ Eq (Star.star (HMul.hMul (HMul.hMul (Star.star c) (Star.star b)) a)) (Star.s …
                                           -/
    _ = star (star c * (star b * a)) := by rw [ha.right_assoc]
                                           /-
                                             🎉 no goals
                                           -/
                                    /-
                                      R : Type u_1
                                      inst✝¹ : Mul R
                                      inst✝ : StarMul R
                                      a : R
                                      ha : Membership.mem (Set.center R) a
                                      b c : R
                                      ⊢ Eq (Star.star (HMul.hMul (Star.star c) (HMul.hMul (Star.star b) a))) (HMul.h …
                                    -/
    _ = star (star b * a) * c := by rw [star_mul, star_star]
                                    /-
                                      🎉 no goals
                                    -/
                               /-
                                 R : Type u_1
                                 inst✝¹ : Mul R
                                 inst✝ : StarMul R
                                 a : R
                                 ha : Membership.mem (Set.center R) a
                                 b c : R
                                 ⊢ Eq (HMul.hMul (Star.star (HMul.hMul (Star.star b) a)) c) (HMul.hMul (HMul.hM …
                               -/
    _ = (star a * b) * c := by rw [star_mul, star_star]
                               /-
                                 🎉 no goals
                               -/
  mid_assoc b c := calc
                                                             /-
                                                               R : Type u_1
                                                               inst✝¹ : Mul R
                                                               inst✝ : StarMul R
                                                               a : R
                                                               ha : Membership.mem (Set.center R) a
                                                               b c : R
                                                               ⊢ Eq (HMul.hMul (HMul.hMul b (Star.star a)) c) (Star.star (HMul.hMul (Star.sta …
                                                             -/
    b * star a * c = star (star c * star (b * star a)) := by rw [← star_mul, star_star]
                                                             /-
                                                               🎉 no goals
                                                             -/
                                           /-
                                             R : Type u_1
                                             inst✝¹ : Mul R
                                             inst✝ : StarMul R
                                             a : R
                                             ha : Membership.mem (Set.center R) a
                                             b c : R
                                             ⊢ Eq (Star.star (HMul.hMul (Star.star c) (Star.star (HMul.hMul b (Star.star a) …
                                           -/
    _ = star (star c * (a * star b)) := by rw [star_mul b, star_star]
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             R : Type u_1
                                             inst✝¹ : Mul R
                                             inst✝ : StarMul R
                                             a : R
                                             ha : Membership.mem (Set.center R) a
                                             b c : R
                                             ⊢ Eq (Star.star (HMul.hMul (Star.star c) (HMul.hMul a (Star.star b)))) (Star.s …
                                           -/
    _ = star ((star c * a) * star b) := by rw [ha.mid_assoc]
                                           /-
                                             🎉 no goals
                                           -/
                               /-
                                 R : Type u_1
                                 inst✝¹ : Mul R
                                 inst✝ : StarMul R
                                 a : R
                                 ha : Membership.mem (Set.center R) a
                                 b c : R
                                 ⊢ Eq (Star.star (HMul.hMul (HMul.hMul (Star.star c) a) (Star.star b))) (HMul.h …
                               -/
    _ = b * (star a * c) := by rw [star_mul, star_star, star_mul (star c), star_star]
                               /-
                                 🎉 no goals
                               -/
  right_assoc b c := calc
                                                   /-
                                                     R : Type u_1
                                                     inst✝¹ : Mul R
                                                     inst✝ : StarMul R
                                                     a : R
                                                     ha : Membership.mem (Set.center R) a
                                                     b c : R
                                                     ⊢ Eq (HMul.hMul (HMul.hMul b c) (Star.star a)) (Star.star (HMul.hMul a (Star.s …
                                                   -/
    b * c * star a = star (a * star (b * c)) := by rw [star_mul, star_star]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                           /-
                                             R : Type u_1
                                             inst✝¹ : Mul R
                                             inst✝ : StarMul R
                                             a : R
                                             ha : Membership.mem (Set.center R) a
                                             b c : R
                                             ⊢ Eq (Star.star (HMul.hMul a (Star.star (HMul.hMul b c)))) (Star.star (HMul.hM …
                                           -/
    _ = star (a * (star c * star b)) := by rw [star_mul b]
                                           /-
                                             🎉 no goals
                                           -/
                                           /-
                                             R : Type u_1
                                             inst✝¹ : Mul R
                                             inst✝ : StarMul R
                                             a : R
                                             ha : Membership.mem (Set.center R) a
                                             b c : R
                                             ⊢ Eq (Star.star (HMul.hMul a (HMul.hMul (Star.star c) (Star.star b)))) (Star.s …
                                           -/
    _ = star ((a * star c) * star b) := by rw [ha.left_assoc]
                                           /-
                                             🎉 no goals
                                           -/
                                    /-
                                      R : Type u_1
                                      inst✝¹ : Mul R
                                      inst✝ : StarMul R
                                      a : R
                                      ha : Membership.mem (Set.center R) a
                                      b c : R
                                      ⊢ Eq (Star.star (HMul.hMul (HMul.hMul a (Star.star c)) (Star.star b))) (HMul.h …
                                    -/
    _ = b * star (a * star c) := by rw [star_mul, star_star]
                                    /-
                                      🎉 no goals
                                    -/
                               /-
                                 R : Type u_1
                                 inst✝¹ : Mul R
                                 inst✝ : StarMul R
                                 a : R
                                 ha : Membership.mem (Set.center R) a
                                 b c : R
                                 ⊢ Eq (HMul.hMul b (Star.star (HMul.hMul a (Star.star c)))) (HMul.hMul b (HMul. …
                               -/
    _ = b * (c * star a) := by rw [star_mul, star_star]
                               /-
                                 🎉 no goals
                               -/


theorem Set.star_centralizer : star s.centralizer = (star s).centralizer := by
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    s : Set R
    ⊢ Eq (Star.star s.centralizer) (Star.star s).centralizer
  -/
  simp_rw [centralizer, ← commute_iff_eq]
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    s : Set R
    ⊢ Eq (Star.star (setOf fun c => ∀ (m : R), Membership.mem s m → Commute m c))  …
  -/
  conv_lhs => simp only [← star_preimage, preimage_setOf_eq, ← commute_star_comm]
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    s : Set R
    ⊢ Eq (setOf fun a => ∀ (m : R), Membership.mem s m → Commute (Star.star m) a)  …
  -/
  conv_rhs => simp only [← image_star, forall_mem_image]
  /-
    🎉 no goals
  -/


theorem Set.union_star_self_comm (hcomm : ∀ x ∈ s, ∀ y ∈ s, y * x = x * y)
    (hcomm_star : ∀ x ∈ s, ∀ y ∈ s, y * star x = star x * y) :
    ∀ x ∈ s ∪ star s, ∀ y ∈ s ∪ star s, y * x = x * y := by
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    s : Set R
    hcomm : ∀ (x : R), Membership.mem s x → ∀ (y : R), Membership.mem s y → Eq (HM …
    hcomm_star : ∀ (x : R), Membership.mem s x → ∀ (y : R), Membership.mem s y → E …
    ⊢ ∀ (x : R), Membership.mem (Union.union s (Star.star s)) x → ∀ (y : R), Membe …
  -/
  change s ∪ star s ⊆ (s ∪ star s).centralizer
  simp_rw [centralizer_union, ← star_centralizer, union_subset_iff, subset_inter_iff,
    star_subset_star, star_subset]
  /-
    R : Type u_1
    inst✝¹ : Mul R
    inst✝ : StarMul R
    s : Set R
    hcomm : ∀ (x : R), Membership.mem s x → ∀ (y : R), Membership.mem s y → Eq (HM …
    hcomm_star : ∀ (x : R), Membership.mem s x → ∀ (y : R), Membership.mem s y → E …
    ⊢ And (And (HasSubset.Subset s s.centralizer) (HasSubset.Subset s (Star.star s …
  -/
  exact ⟨⟨hcomm, hcomm_star⟩, ⟨hcomm_star, hcomm⟩⟩
  /-
    🎉 no goals
  -/


theorem Set.star_mem_centralizer' (h : ∀ a : R, a ∈ s → star a ∈ s) (ha : a ∈ Set.centralizer s) :
                                                 /-
                                                   R : Type u_1
                                                   inst✝¹ : Mul R
                                                   inst✝ : StarMul R
                                                   a : R
                                                   s : Set R
                                                   h : ∀ (a : R), Membership.mem s a → Membership.mem s (Star.star a)
                                                   ha : Membership.mem s.centralizer a
                                                   y : R
                                                   hy : Membership.mem s y
                                                   ⊢ Eq (HMul.hMul y (Star.star a)) (HMul.hMul (Star.star a) y)
                                                 -/
    star a ∈ Set.centralizer s := fun y hy => by simpa using congr_arg star (ha _ (h _ hy)).symm
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem Set.star_mem_centralizer (ha : a ∈ Set.centralizer (s ∪ star s)) :
    star a ∈ Set.centralizer (s ∪ star s) :=
  Set.star_mem_centralizer'
    (fun _x hx => hx.elim (fun hx => Or.inr <| Set.star_mem_star.mpr hx) Or.inl) ha

